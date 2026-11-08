# [preannotation] - START
"""Optional real-browser regression; no browser dependency in the core suite."""
import json
import os
import subprocess
import sys

import pytest

from nnslr_tools.review import write_review


def test_browser_preserves_proposals_exports_confirmed_and_reports_storage_failure(tmp_path):
    python = os.environ.get('NNSLR_TEST_BROWSER_PYTHON', sys.executable)
    chromium = os.environ.get('NNSLR_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('optional browser environment not configured')
    check = subprocess.run([python,'-c','import playwright.sync_api'],capture_output=True)
    if check.returncode:
        pytest.skip('optional Playwright environment not configured')
    proposal = {'bbox_xyxy':[20,40,70,50], 'family':'road_marking_candidate',
                'value_kph':30,'score':.95,'model_label':'30','detection_id':'0/0/0'}
    rows = [{'frame_key':f'0/{i}','segment_index':0,'output_index':i,
             'width':200,'height':100,'preview_path':'images/00000000.jpg',
             'detections':[proposal] if i==0 else []} for i in range(2)]
    write_review(tmp_path,rows,'synthetic-browser-test')
    code = r'''
import functools,json,sys,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
root=Path(sys.argv[1])
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(root)))
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=sys.argv[2],headless=True)
        page=browser.new_page();errors=[]
        page.on('pageerror',lambda err:errors.append(str(err)))
        page.goto(f'http://127.0.0.1:{server.server_port}')
        page.locator('#reviewer').fill('synthetic-human')
        page.locator('#boxList .box-main').click()
        assert page.locator('#familiesMore [data-family="road_marking_candidate"]').get_attribute('aria-pressed')=='true'
        page.locator('#value').fill('50');page.locator('#value').dispatch_event('change')
        page.locator('#confirm').click();page.reload()
        # The confirmation moved on to the next frame: back to 0/0 with the drive's bar.
        page.locator('#scrubber').evaluate("(el) => { el.value = '0'; el.dispatchEvent(new Event('input', {bubbles: true})); }")
        page.wait_for_function("document.querySelector('#confirm').innerText.includes('Confirmed')")
        assert 'Confirmed' in page.locator('#confirm').inner_text()
        page.locator('#boxList .box-main').click()
        assert page.locator('#value').input_value()=='50'
        page.locator('summary', has_text='More').click()
        with page.expect_download() as event:page.locator('#export').click()
        event.value.save_as(root/'reviewed.jsonl')
        exported=[json.loads(s) for s in (root/'reviewed.jsonl').read_text().splitlines()]
        assert len(exported)==1 and exported[0]['frame_key']=='0/0'
        assert exported[0]['detections'][0]['value_kph']==50
        assert exported[0]['source_proposals'][0]['value_kph']==30
        assert exported[0]['reviewer']=='synthetic-human'
        assert exported[0]['review_timestamp'].endswith('Z')
        assert not exported[0]['training_ready'] and not exported[0]['ground_truth']
        page.evaluate("() => { Storage.prototype.setItem=()=>{throw new DOMException('full','QuotaExceededError')}; }")
        page.locator('#value').fill('30');page.locator('#value').dispatch_event('change')
        assert page.locator('#confirm').inner_text().startswith('Confirm ·')
        assert 'Saving in this browser is not available' in page.locator('#notices').inner_text()
        page.locator('#next').click()
        assert 'Saving in this browser is not available' in page.locator('#notices').inner_text()
        assert not errors,errors
        browser.close()
finally:
    server.shutdown()
'''
    result = subprocess.run([python,'-c',code,str(tmp_path),chromium],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
# [preannotation] - END
