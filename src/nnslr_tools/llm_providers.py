"""The LLM providers that can review signs: OpenAI-compatible servers (strata, a llama.cpp server with
Qwen3-VL, any other /v1 endpoint), with their address, model and API key, and the one that reviews.

They are kept in one JSON file, ``<data root>/web/private/llm-providers.json`` (folder 700, file 600, written
all at once by review_auth.write_private), or in the file named by NNSLR_LLM_PROVIDERS. The data root is under
data/, which git ignores. Settings on the site and ``nnslr llm-provider`` write it; the AI review jobs read it.

A key never leaves the server: every answer about a provider is public_provider(), which only says has_key, and
every error message replaces any copy of the key with ***. The HTTP client refuses redirects (one could carry
the key to another address), reads at most 2 MB and turns every failure into an LlmError whose message is safe
to show. Standard library only.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from nnslr_tools.review_auth import read_private, write_private
from nnslr_tools.review_db import ReviewRejected

PROVIDERS_ENV = 'NNSLR_LLM_PROVIDERS'
SCHEMA_VERSION = 1
MAX_PROVIDERS = 20
MODELS_TIMEOUT = 15.0
MAX_ANSWER_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_MODELS = 500
ID_PATTERN = re.compile(r'[a-z][a-z0-9-]{0,31}')
# What may stay of a model name in a reviewer name or a file name.
SLUG_UNSAFE = re.compile(r'[^A-Za-z0-9._:+-]+')
# The reviewer name is stored as "ai/<id>/<model>", at most review_db.MAX_REVIEWER_LENGTH (100) characters.
MAX_REVIEWER_NAME = 97
TEST_FIELDS = ('ok', 'answered_json', 'sees_images', 'seconds', 'message', 'frame', 'at')
# Every read-modify-write of the file holds it.
_LOCK = threading.Lock()


class LlmError(ReviewRejected):
    """A call to an AI server that failed: status 502 (ai_unreachable, ai_http_error, ai_bad_answer,
    ai_redirect) or 504 (ai_timeout). The message is safe to show: the key masked, the provider's text
    cleaned, at most 200 characters. ``http_code`` is the provider's HTTP status for ai_http_error."""

    def __init__(self, status: int, reason: str, message: str, *, http_code: int | None = None):
        super().__init__(status, reason, _clean(message, 200))
        self.http_code = http_code


# An answer cut at max_tokens: the server is up, the answer did not fit (Strata says it with HTTP 502, "structured
# output was incomplete (finish_reason=length); increase the output budget"; others with an empty answer).
INCOMPLETE = re.compile(r'finish_reason[ =:]*length|output budget', re.I)


def incomplete(exc: ReviewRejected) -> bool:
    """The AI server answered, but its answer was cut at max_tokens: ask again with more, it is not down."""
    return exc.reason in ('ai_http_error', 'ai_bad_answer') and bool(INCOMPLETE.search(exc.message or ''))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _clean(text, limit: int) -> str:
    """One line of printable text, at most ``limit`` characters."""
    text = ' '.join(''.join(character if character.isprintable() else ' ' for character in str(text)).split())
    if len(text) > limit:
        text = text[:limit - 1] + '…'
    return text


def _masked(text: str, key: str | None) -> str:
    """The text with the key, and any long piece of it, replaced by ***."""
    if not key:
        return text
    text = text.replace(key, '***')

    def piece(match):
        if match.group(0) in key:
            return '***'
        return match.group(0)
    return re.sub(r'[A-Za-z0-9._~+/=-]{8,}', piece, text)


def _invalid(message: str) -> ReviewRejected:
    return ReviewRejected(400, 'invalid_provider', message)


# ---------- the file ----------

def providers_path(root: Path) -> Path:
    explicit = os.environ.get(PROVIDERS_ENV, '').strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path(root) / 'web' / 'private' / 'llm-providers.json'


def _empty() -> dict:
    return {'schema_version': SCHEMA_VERSION, 'reviewer': None, 'reviewer_off': False, 'providers': []}


def _load(root: Path) -> dict:
    path = providers_path(root)
    try:
        if path.stat().st_mode & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass
    text = read_private(path)
    if not text.strip():
        return _empty()
    try:
        data = json.loads(text)
    except ValueError:
        raise ReviewRejected(500, 'unreadable', 'The AI provider file on this server is not valid JSON: '
                                                'fix it or move it away.') from None
    if not isinstance(data, dict) or not isinstance(data.get('providers'), list):
        raise ReviewRejected(500, 'unreadable', 'The AI provider file on this server has no provider list: '
                                                'fix it or move it away.')
    providers = []
    for item in data['providers']:
        if isinstance(item, dict) and isinstance(item.get('id'), str) and ID_PATTERN.fullmatch(item['id']) \
                and all(isinstance(item.get(name), str) for name in ('name', 'base_url', 'model')):
            providers.append(item)
    reviewer = data.get('reviewer')
    if not any(item['id'] == reviewer for item in providers):
        reviewer = None
    # "Don't use AI" chosen in Settings: a Test that passes then does not choose its provider by itself.
    off = reviewer is None and data.get('reviewer_off') is True
    return {'schema_version': SCHEMA_VERSION, 'reviewer': reviewer, 'reviewer_off': off, 'providers': providers}


def _store(root: Path, data: dict) -> None:
    path = providers_path(root)
    write_private(path, json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    if not os.environ.get(PROVIDERS_ENV, '').strip():
        # Our own folder (an explicit file may live in a folder shared with other things).
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass


def _find(data: dict, provider_id) -> dict | None:
    for item in data['providers']:
        if item['id'] == provider_id:
            return item
    return None


def _unknown() -> ReviewRejected:
    return ReviewRejected(404, 'unknown_provider', 'Unknown AI provider: it may have been removed.')


# ---------- names ----------

def _slug(model: str) -> str:
    return SLUG_UNSAFE.sub('-', model).strip('-') or 'model'


def model_slug(model: str) -> str:
    """The model in a file name: letters, digits and ._:+- only, at most 80 characters."""
    return _slug(model)[:80]


def reviewer_name(record: dict) -> str:
    """'<id>/<model>' (without 'ai/'): the reviewer of the AI reviews this provider saves."""
    return (record['id'] + '/' + _slug(record['model']))[:MAX_REVIEWER_NAME]


def id_from_name(name: str) -> str:
    """The id a new provider of this name gets, before collisions are resolved."""
    text = re.sub(r'[^a-z0-9]+', '-', name.strip().lower()).strip('-')[:32].strip('-')
    if not text:
        return 'provider'
    if not text[0].isalpha():
        text = ('p-' + text)[:32].strip('-')
    return text


def _new_id(name: str, taken: set) -> str:
    base = id_from_name(name)
    candidate = base
    number = 1
    while candidate in taken:
        number = number + 1
        suffix = f'-{number}'
        candidate = base[:32 - len(suffix)].rstrip('-') + suffix
    return candidate


# ---------- validation ----------

def _check_name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid('name: give the provider a name.')
    value = value.strip()
    if len(value) > 40 or not value.isprintable():
        raise _invalid('name: at most 40 printable characters.')
    return value


def check_base_url(value) -> str:
    """The address of an OpenAI-compatible API, normalized: http(s)://host[:port]/.../v1."""
    if not isinstance(value, str) or not value.strip():
        raise _invalid('base_url: give the address of the server, ending in /v1.')
    text = value.strip()
    if len(text) > 200 or any(not character.isprintable() or character.isspace() for character in text):
        raise _invalid('base_url: at most 200 characters, without spaces.')
    try:
        parts = urllib.parse.urlsplit(text)
        port = parts.port
    except ValueError:
        raise _invalid('base_url: not a valid address.') from None
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise _invalid('base_url: it must start with http:// or https:// and name a server.')
    if '@' in parts.netloc:
        raise _invalid('base_url: no user name or password in the address: give the key in its own field.')
    if '?' in text or '#' in text:
        raise _invalid('base_url: no query or fragment (? or #) in the address.')
    path = parts.path.rstrip('/')
    if not path.endswith('/v1'):
        raise _invalid('base_url: the path must end with /v1, e.g. https://host/v1.')
    netloc = parts.hostname if ':' not in parts.hostname else '[' + parts.hostname + ']'
    if port is not None:
        netloc = netloc + ':' + str(port)
    return parts.scheme + '://' + netloc + path


def _check_model(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid('model: give the model name the server knows.')
    value = value.strip()
    if len(value) > 200 or any(not character.isprintable() or character.isspace() for character in value):
        raise _invalid('model: at most 200 printable characters, without spaces.')
    return value


def _check_key(value) -> str:
    if not isinstance(value, str):
        raise _invalid('key: text, or empty to remove it.')
    text = value.strip()
    if not text or len(text) > 512 or any(not 33 <= ord(character) <= 126 for character in text):
        raise _invalid('key: 1-512 printable ASCII characters without spaces.')
    return text


# ---------- what the site sees ----------

def public_provider(record: dict, reviewer) -> dict:
    """A provider as every answer shows it: never its key, only whether it has one."""
    last_test = record.get('last_test')
    return {'id': record['id'], 'name': record['name'], 'base_url': record['base_url'], 'model': record['model'],
            'has_key': bool(record.get('key')), 'is_reviewer': record['id'] == reviewer,
            'reviewer_name': 'ai/' + reviewer_name(record), 'updated_at': record.get('updated_at'),
            'last_test': copy.deepcopy(last_test) if isinstance(last_test, dict) else None}


def public_settings(root: Path) -> dict:
    data = _load(root)
    providers = []
    for item in data['providers']:
        providers.append(public_provider(item, data['reviewer']))
    return {'providers': providers, 'reviewer': data['reviewer']}


def provider(root: Path, provider_id) -> dict:
    """The private record of a provider (with 'key' when it has one): for this server's own calls only."""
    record = _find(_load(root), provider_id)
    if record is None:
        raise _unknown()
    return copy.deepcopy(record)


def reviewer_provider(root: Path) -> dict | None:
    data = _load(root)
    record = _find(data, data['reviewer'])
    if record is None:
        return None
    return copy.deepcopy(record)


def review_refusal(record: dict) -> str:
    """Why this provider may not review ('' when it may). Only a provider whose last Test passed reviews: a model
    that cannot see the frames, or does not answer, must never save final reviews. A new address or model clears
    the Test (save_provider), so it has to pass again."""
    test = record.get('last_test')
    if not isinstance(test, dict):
        return f'Test {record["name"]} in Settings first: only a provider that passed its Test can review.'
    if test.get('ok') is not True:
        return f'{record["name"]} did not pass its last Test. Fix it, then test it again in Settings.'
    return ''


def check_can_review(record: dict) -> None:
    """ReviewRejected 409 ai_not_tested unless the provider passed its last Test (review_refusal)."""
    refusal = review_refusal(record)
    if refusal:
        raise ReviewRejected(409, 'ai_not_tested', refusal)


def ai_state(root: Path) -> dict:
    """Whether an AI reviewer is chosen and passed its Test (a file read, no network): {available, reason,
    provider}. A reviewer that did not pass is named in provider, with available False and the reason."""
    try:
        record = reviewer_provider(root)
    except ReviewRejected as exc:
        return {'available': False, 'reason': exc.message, 'provider': None}
    if record is None:
        return {'available': False, 'reason': 'Choose an AI reviewer in Settings.', 'provider': None}
    refusal = review_refusal(record)
    return {'available': not refusal, 'reason': refusal,
            'provider': {'id': record['id'], 'name': record['name'], 'model': record['model'],
                         'reviewer_name': 'ai/' + reviewer_name(record)}}


# ---------- changes ----------

def save_provider(root: Path, body) -> dict:
    """Add a provider (no id) or change one ({id, ...}). key: absent keeps it, "" removes it, text sets it."""
    if not isinstance(body, dict):
        raise _invalid('the request must be an object.')
    name = _check_name(body.get('name'))
    base_url = check_base_url(body.get('base_url'))
    model = _check_model(body.get('model'))
    key = None
    if 'key' in body and body['key'] is not None and body['key'] != '':
        key = _check_key(body['key'])
    with _LOCK:
        data = _load(root)
        now = _now()
        provider_id = body.get('id')
        if provider_id is None or provider_id == '':
            if len(data['providers']) >= MAX_PROVIDERS:
                raise ReviewRejected(400, 'too_many_providers',
                                     f'At most {MAX_PROVIDERS} AI providers: remove one first.')
            taken = {item['id'] for item in data['providers']}
            record = {'id': _new_id(name, taken), 'name': name, 'base_url': base_url, 'model': model,
                      'created_at': now, 'updated_at': now, 'last_test': None}
            if key is not None:
                record['key'] = key
            data['providers'].append(record)
        else:
            record = _find(data, provider_id)
            if record is None:
                raise _unknown()
            if record['base_url'] != base_url or record['model'] != model:
                record['last_test'] = None
            record.update(name=name, base_url=base_url, model=model, updated_at=now)
            if key is not None:
                record['key'] = key
            elif body.get('key') == '':
                record.pop('key', None)
        _store(root, data)
        return public_provider(record, data['reviewer'])


def delete_provider(root: Path, provider_id) -> None:
    with _LOCK:
        data = _load(root)
        record = _find(data, provider_id)
        if record is None:
            raise _unknown()
        data['providers'].remove(record)
        if data['reviewer'] == provider_id:
            # Another provider that passed its Test reviews instead, when there is one.
            data['reviewer'] = next((item['id'] for item in data['providers']
                                     if (item.get('last_test') or {}).get('ok') is True), None)
        _store(root, data)


def set_reviewer(root: Path, provider_id) -> None:
    """The provider that reviews (the site's Ask AI, the AI review jobs), or None for none. Only one that passed
    its last Test (check_can_review)."""
    with _LOCK:
        data = _load(root)
        if provider_id is not None:
            record = _find(data, provider_id)
            if record is None:
                raise _unknown()
            check_can_review(record)
        data['reviewer'] = provider_id
        data['reviewer_off'] = provider_id is None
        _store(root, data)


def record_test(root: Path, provider_id: str, result: dict) -> None:
    """Keep the last Test of a provider (gone with it if it was removed in the meantime). One that passes reviews
    when none does, unless "Don't use AI" was chosen: the first provider that works is the reviewer by itself."""
    with _LOCK:
        data = _load(root)
        record = _find(data, provider_id)
        if record is None:
            return
        record['last_test'] = {name: copy.deepcopy(result.get(name)) for name in TEST_FIELDS}
        if result.get('ok') is True and data['reviewer'] is None and not data['reviewer_off']:
            data['reviewer'] = provider_id
        _store(root, data)


# ---------- HTTP ----------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect could carry the key to another address: refuse every one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LlmError(502, 'ai_redirect', f'The AI server answered with a redirect (HTTP {code}): '
                                           'give its final address.')


_OPENER = urllib.request.build_opener(_NoRedirect)


def _provider_message(error: urllib.error.HTTPError) -> str:
    """error.message (or error, message, detail) of the provider's JSON answer, '' without one."""
    try:
        raw = error.read(MAX_ERROR_BYTES)
        data = json.loads(raw.decode('utf-8', errors='replace'))
    except (OSError, ValueError, http.client.HTTPException):
        return ''
    if not isinstance(data, dict):
        return ''
    found = data.get('error')
    if isinstance(found, dict):
        found = found.get('message')
    if not isinstance(found, str):
        found = data.get('message') if isinstance(data.get('message'), str) else data.get('detail')
    return found if isinstance(found, str) else ''


def _call(url: str, key: str | None, *, body: dict | None = None, timeout: float) -> object:
    """The JSON answer of a GET (no body) or POST to an OpenAI-compatible server."""
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'speedlimits-web'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    data = None
    if body is not None:
        data = json.dumps(body, allow_nan=False).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers=headers, method='GET' if body is None else 'POST')
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_ANSWER_BYTES + 1)
    except LlmError:
        raise
    except urllib.error.HTTPError as exc:
        detail = _clean(_masked(_provider_message(exc), key), 160)
        exc.close()
        if not detail and exc.code in (502, 503, 504):
            # The proxy in front of the AI server answers, the server behind it does not.
            detail = 'the AI server ' + (urllib.parse.urlsplit(url).hostname or '') + ' is down or restarting, try later'
        raise LlmError(502, 'ai_http_error', _masked(f'HTTP {exc.code}' + (': ' + detail if detail else ''), key),
                       http_code=exc.code) from None
    except (socket.timeout, TimeoutError):
        raise LlmError(504, 'ai_timeout', f'The AI server did not answer within {timeout:g} s.') from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise LlmError(504, 'ai_timeout', f'The AI server did not answer within {timeout:g} s.') from None
        reason = exc.reason if isinstance(exc.reason, str) else type(exc.reason).__name__ + ' ' + str(exc.reason)
        raise LlmError(502, 'ai_unreachable', _masked('Cannot reach the AI server (' + _clean(reason, 120) + ').',
                                                      key)) from None
    except (http.client.HTTPException, OSError, ValueError) as exc:
        raise LlmError(502, 'ai_unreachable', _masked('Cannot reach the AI server (' + type(exc).__name__ + ').',
                                                      key)) from None
    if len(raw) > MAX_ANSWER_BYTES:
        raise LlmError(502, 'ai_bad_answer', 'The AI server answered with more than 2 MB.')
    try:
        return json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise LlmError(502, 'ai_bad_answer', 'The AI server did not answer with JSON.') from None


def _models_of(answer) -> list[dict]:
    items = None
    if isinstance(answer, dict):
        items = answer.get('data')
        if not isinstance(items, list):
            items = answer.get('models')
    if not isinstance(items, list):
        raise LlmError(502, 'ai_bad_answer', 'The AI server did not list its models.')
    models = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get('id')
        if not isinstance(name, str):
            name = item.get('model') if isinstance(item.get('model'), str) else item.get('name')
        if not isinstance(name, str) or not name.strip() or name in seen:
            continue
        name = _clean(name, 200)
        seen.add(name)
        sees_images = None
        architecture = item.get('architecture')
        if isinstance(architecture, dict) and isinstance(architecture.get('input_modalities'), list):
            sees_images = 'image' in architecture['input_modalities']
        models.append({'id': name, 'sees_images': sees_images})
        if len(models) == MAX_MODELS:
            break
    return models


def list_models(root: Path, body) -> list[dict]:
    """GET {base_url}/models: [{id, sees_images}]. body {id?, base_url?, key?}.

    The key is body.key, else the stored key of ``id`` but only for the stored address: a stored key is never
    sent to an address someone just typed."""
    if not isinstance(body, dict):
        raise _invalid('the request must be an object.')
    stored = None
    if body.get('id') not in (None, ''):
        stored = provider(root, body['id'])
    if body.get('base_url') in (None, ''):
        if stored is None:
            raise _invalid('base_url: give the address of the server, ending in /v1.')
        base_url = stored['base_url']
    else:
        base_url = check_base_url(body['base_url'])
    key = None
    if isinstance(body.get('key'), str) and body['key'] != '':
        key = _check_key(body['key'])
    elif stored is not None and stored['base_url'] == base_url:
        key = stored.get('key')
    return _models_of(_call(base_url + '/models', key, timeout=MODELS_TIMEOUT))


def _content(answer) -> str:
    try:
        choice = answer['choices'][0]
        content = choice['message']['content']
    except (KeyError, IndexError, TypeError):
        raise LlmError(502, 'ai_bad_answer', 'The AI server answered without a message.') from None
    if isinstance(content, list):
        # Some servers answer in parts: [{type: text, text}].
        content = ''.join(part.get('text', '') for part in content
                          if isinstance(part, dict) and isinstance(part.get('text'), str))
    if not isinstance(content, str):
        raise LlmError(502, 'ai_bad_answer', 'The AI server answered without a message.')
    if not content.strip():
        finish = choice.get('finish_reason') if isinstance(choice, dict) else None
        raise LlmError(502, 'ai_bad_answer', 'The AI server gave an empty answer'
                       + (f' (finish_reason {_clean(finish, 40)})' if isinstance(finish, str) else '') + '.')
    return content


# Thinking models (Qwen3, Qwen3.8) think before they answer: on a frame Qwen3.8-27B on llama.cpp took 84 s and spent
# every token of max_tokens thinking, with no answer. The servers of open models (llama.cpp, vLLM, SGLang) take
# chat_template_kwargs to turn it off; one that does not know it (HTTP 400, e.g. OpenAI) is asked again without.
NO_THINKING = {'enable_thinking': False}
OPTIONAL_FIELDS = ('chat_template_kwargs', 'response_format')


def chat(record: dict, messages: list[dict], *, timeout: float, json_mode: bool = True,
         max_tokens: int = 2048) -> tuple[str, float]:
    """(assistant text, seconds) of POST {base_url}/chat/completions, temperature 0, without thinking.

    With json_mode the answer is asked as a JSON object. A server that refuses a request (HTTP 400) is asked again
    without chat_template_kwargs, then without response_format, then without both."""
    body = {'model': record['model'], 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens,
            'chat_template_kwargs': dict(NO_THINKING)}
    if json_mode:
        body['response_format'] = {'type': 'json_object'}
    optional = [name for name in OPTIONAL_FIELDS if name in body]
    # The full request, then each optional field left out alone, then all of them.
    attempts = [()] + [(name,) for name in optional] + ([tuple(optional)] if len(optional) > 1 else [])
    url = record['base_url'] + '/chat/completions'
    started = time.monotonic()
    for number, left_out in enumerate(attempts):
        try:
            answer = _call(url, record.get('key'), body={key: value for key, value in body.items()
                                                         if key not in left_out}, timeout=timeout)
            break
        except LlmError as exc:
            if exc.http_code != 400 or number == len(attempts) - 1:
                raise
    return _content(answer), time.monotonic() - started
