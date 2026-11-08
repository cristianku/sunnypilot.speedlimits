# Brain training diagram — simplify the portal UI

## Task

Redesign **Training → Brain**'s explanatory diagram for mobile first. This is a **UI/documentation-only** change: **do not modify neural-network architecture, training, datasets, inference, or model artifacts**.

Use a short, vertically flowing diagram that makes the real data path obvious:

```text
INPUTS · 4 times/second
  Road image (256 × 144) ─→ MobileNetV3-small ─┐
  Read signs + plates + sign distance ─────────┤
  Hour + country + lead-car information ───────┘
                                               ↓
                                           GRU memory
                                               ↓
                              ┌────────────────┴───────────────┐
                              ↓                                ↓
                       Speed to hold                 Next limit + metres
```

The visual layout must show **MobileNetV3 before the GRU**, not as two parallel modules. Other inputs join the **visual features before the GRU**; they do not pass through MobileNetV3. Check actual selected groups against the current brain contract instead of hard-coding outdated input claims. Pedals and car speed are **not** default inference inputs (`driver=False`).

Under the main flow, add one compact **ONLY DURING TRAINING** section:

- **Road type** — an auxiliary prediction taught by offline map labels, corrected in Speed Review.
- **Roadworks** — an auxiliary prediction taught by Speed Review labels.
- **Road characteristics** — auxiliary predictions from reviewed road-kind labels.
- **Correct speed** — the training target from the recorded driver's speed, corrected/filtered in Speed Review (not an input).
- **Next limit / metres** — training targets from reviewed/read sign events (the corresponding predictions *are* outputs on the car).

Explain in a single sentence: **“Extra labels teach the shared network during training; they are not inputs and are not needed in the car.”**

### Presentation rules

- **Blue:** inputs. **Green:** outputs available on the car. **Orange, dashed:** training-only predictions/labels.
- Keep the graphic **small, legible and responsive**, especially in a single column on a phone; no sprawling boxes or duplicated explanations.
- Use plain English labels in the diagram; keep internal names such as `type_head`, `road_head` and `works_head` out of the visible UI.
- Make the difference between **inputs**, **predictions** and **training targets** unambiguous.
- The map may create **offline supervision labels only**. **No map fields enter the Brain at inference.**
- Do not imply that auxiliary outputs feed the speed prediction directly: they influence the shared encoder/GRU weights **through their training losses**.
- Preserve existing site styling and interactions; update any UI tests needed by the presentation change.

**Source of truth:** `src/nnslr_tools/brain_net.py`, `brain_train.py`, `brain_targets.py` and the exported Brain contract. If this document conflicts with the code, follow the code and correct the UI text.
