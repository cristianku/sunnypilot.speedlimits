"""The last step from a label to the weights: which frames give a gradient, and where it goes. These need torch, which only
.venv-gpu has, and .venv-gpu has no pytest: they are skipped in .venv. Small pictures on the CPU, no training:
uv pip install --target /tmp/pytest-gpu --python .venv-gpu/bin/python pytest
PYTHONPATH=/tmp/pytest-gpu .venv-gpu/bin/python -m pytest -o addopts="" -p no:cacheprovider tests/dataset/test_brain_net.py"""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')
SIZE = 0
T = 5


def setup_module():
    global SIZE
    from nnslr_tools.brain_targets import INPUT_SIZES
    SIZE = sum(INPUT_SIZES[name] for name in GROUPS)


def net(**options):
    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(GROUPS, pretrained=False, **options)
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    return model


def batch(t=T):
    torch.manual_seed(1)
    return torch.rand(1, t, 3, 144, 256), torch.rand(1, t, SIZE)


def answers(t=T, speed_on=(), works_on=()):
    speed = torch.full((1, t), 19, dtype=torch.long)
    speed_weight, works_weight = torch.zeros(1, t), torch.zeros(1, t)
    speed_weight[0, list(speed_on)] = 1
    works_weight[0, list(works_on)] = 1
    return speed, speed_weight, torch.ones(1, t), works_weight


def gradient_of(model, part):
    found = [value.grad for name, value in model.named_parameters() if name.startswith(part) and value.grad is not None]
    return sum(float(grad.abs().sum()) for grad in found)


def backward(model, images, extra, labels):
    from nnslr_tools.brain_net import brain_loss
    model.zero_grad()
    speed_logits, works_logit, _state = model(images, extra)
    speed, speed_weight, works, works_weight = labels
    loss = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    loss.backward()
    return loss


def test_the_network_says_a_speed_and_works_on_every_frame():
    from nnslr_tools.brain_targets import SPEED_CLASSES
    speed, works, state = net()(*batch())
    assert speed.shape == (1, T, SPEED_CLASSES)
    assert works.shape == (1, T)
    assert state.shape == (1, 128)


@pytest.mark.parametrize('pool', ['mean', 'grid'])
def test_a_mark_reaches_the_memory_and_the_picture_features_and_not_the_speed_head(pool):
    model = net(pool=pool)
    backward(model, *batch(), answers(works_on=[1, 2]))
    assert gradient_of(model, 'works_head') > 0
    assert gradient_of(model, 'gru') > 0
    assert gradient_of(model, 'project') > 0
    assert gradient_of(model, 'speed_head') == 0
    assert gradient_of(model, 'trust') == 0


def test_one_step_on_a_mark_changes_what_the_network_says_of_the_speed():
    model = net()
    images, extra = batch()
    with torch.no_grad():
        before, _works, _state = model(images, extra)
    head = model.speed_head.weight.clone()
    backward(model, images, extra, answers(works_on=[1, 2]))
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    with torch.no_grad():
        after, _works, _state = model(images, extra)
    assert torch.equal(head, model.speed_head.weight)
    assert not torch.equal(before, after)


def test_a_stretch_with_no_mark_gives_no_gradient_at_all():
    model = net()
    loss = backward(model, *batch(), answers())
    assert float(loss) == 0
    for part in ('works_head', 'speed_head', 'gru', 'project', 'backbone'):
        assert gradient_of(model, part) == 0, part


def test_a_frame_out_of_the_loss_counts_for_nothing_whatever_its_answer_says():
    from nnslr_tools.brain_net import brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0], works_on=[0])
    other, other_works = speed.clone(), works.clone()
    other[0, 3], other_works[0, 3] = 3, 0
    assert float(brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)) == pytest.approx(
        float(brain_loss(speed_logits, works_logit, other, speed_weight, other_works, works_weight)))


def test_the_works_error_is_a_share_of_the_loss():
    from nnslr_tools.brain_net import WORKS_LOSS, brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0, 1], works_on=[2])
    both = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight)
    alone = brain_loss(speed_logits, None, speed, speed_weight, works, works_weight)
    none = brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, works_share=0.0)
    assert float(none) == pytest.approx(float(alone))
    assert float(both) > float(alone)
    assert WORKS_LOSS == pytest.approx(0.3)


def test_with_the_frames_of_a_piece_an_answer_weighs_the_same_in_a_piece_with_few():
    from nnslr_tools.brain_net import brain_loss
    speed_logits, works_logit, _state = net()(*batch())
    speed, speed_weight, works, works_weight = answers(speed_on=[0, 1, 2, 3], works_on=[0])
    alone = speed_weight.clone()
    alone[0, 1:] = 0
    both = brain_loss(speed_logits, None, speed, speed_weight, works, works_weight, frames=T)
    first = brain_loss(speed_logits, None, speed, alone, works, works_weight, frames=T)
    assert float(first) < float(both)
    errors = torch.nn.functional.cross_entropy(speed_logits[0], speed[0], reduction='none')
    assert float(first) == pytest.approx(float(errors[0]) / T)


@pytest.mark.parametrize('pool', ['mean', 'grid'])
def test_a_frame_never_sees_the_frames_after_it_also_in_training(pool):
    model = net(pool=pool)
    images, extra = batch()
    changed, changed_extra = images.clone(), extra.clone()
    changed[0, 3:] = torch.rand_like(changed[0, 3:])
    changed_extra[0, 3:] = 0
    with torch.no_grad():
        speed, works, _state = model(images, extra)
        other, other_works, _state = model(changed, changed_extra)
    assert torch.equal(speed[0, :3], other[0, :3])
    assert torch.equal(works[0, :3], other_works[0, :3])
    assert not torch.equal(speed[0, 3:], other[0, 3:])


def test_a_training_step_leaves_the_encoder_as_it_is():
    model = net()
    before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1.0)
    backward(model, *batch(), answers(speed_on=[1, 2], works_on=[1, 2]))
    optimizer.step()
    assert gradient_of(model, 'backbone') == 0
    assert all(torch.equal(value, before[name]) for name, value in model.backbone.state_dict().items())


def test_the_scale_of_the_picture_does_not_reach_the_memory():
    model = net()
    model.eval()
    images, extra = batch()
    with torch.no_grad():
        before, _works, _state = model(images, extra)
        model.project[0].weight.mul_(1000)
        model.project[0].bias.mul_(1000)
        after, _works, _state = model(images, extra)
    assert float((before - after).abs().max()) < 1e-2


def test_training_does_not_move_the_batch_norm():
    model = net()
    before = {name: value.clone() for name, value in model.state_dict().items() if 'running_' in name}
    with torch.no_grad():
        model(*batch())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_pieces_with_the_state_carried_are_one_whole_pass():
    from nnslr_tools.brain_targets import drive_chunks
    model = net()
    images, extra = batch()
    with torch.no_grad():
        whole, _works, _state = model(images, extra)
        state, pieces = None, []
        for first, last in drive_chunks(T, 2, 1):
            part, _works, state = model(images[:, first:last], extra[:, first:last], state)
            pieces.append(part)
    assert torch.allclose(torch.cat(pieces, dim=1), whole, atol=1e-5)


def test_the_speed_does_not_depend_on_the_works_head():
    model = net()
    images, extra = batch()
    with torch.no_grad():
        speed, _works, _state = model(images, extra)
        model.works_head.weight.mul_(100).add_(1)
        again, _works, _state = model(images, extra)
    assert torch.equal(speed, again)


def test_the_car_gets_everything_but_the_works_head():
    from nnslr_tools.brain_net import build_brain_net, car_weights
    model = net()
    weights = car_weights(model)
    assert not [name for name in weights if name.startswith('works_head')]
    without = build_brain_net(GROUPS, pretrained=False, works=False)
    assert set(weights) == set(without.state_dict())
    without.load_state_dict(weights)


def test_the_comparison_without_the_works_task_has_no_works_output():
    model = net(works=False)
    _speed, works, _state = model(*batch())
    assert works is None
    backward(model, *batch(), answers(speed_on=[1]))
    assert gradient_of(model, 'gru') > 0


def test_a_network_with_another_input_is_the_same_network():
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import INPUT_SIZES
    groups = tuple(INPUT_SIZES)
    torch.manual_seed(0)
    model = build_brain_net(groups, pretrained=False)
    speed, _works, _state = model(torch.rand(1, 2, 3, 144, 256), torch.rand(1, 2, sum(INPUT_SIZES.values())))
    assert speed.shape[1] == 2


def number_row(kph, known=1.0, ended=0.0, groups=GROUPS):
    from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES
    at = sum(INPUT_SIZES[name] for name in groups[:groups.index('signs')])
    extra = torch.zeros(1, 2, sum(INPUT_SIZES[name] for name in groups))
    extra[..., at], extra[..., at + 1], extra[..., at + 6] = kph / BRAIN_INPUTS.SIGN_KPH_SCALE, known, ended
    return extra


def trusting(model, trust, head=None):
    """A network whose only say is a trust of this much and, with head, one class of its own."""
    with torch.no_grad():
        for layer in (model.speed_head, model.trust):
            layer.weight.zero_()
            layer.bias.zero_()
        model.trust.bias.fill_(trust)
        if head is not None:
            model.speed_head.bias[head] = 1
    return model


@pytest.mark.parametrize('kph', [30, 50, 60, 100, 120, 130])
def test_the_number_in_force_goes_to_its_class_alone_as_much_as_the_network_trusts_it(kph):
    from nnslr_tools.brain_targets import speed_class
    speed = trusting(net(), 3.0)(batch(2)[0], number_row(kph))[0]
    assert speed.argmax(dim=-1).tolist() == [[speed_class(kph)] * 2]
    assert speed.sum(dim=-1).tolist() == [[3.0, 3.0]]


@pytest.mark.parametrize('changes', [{'kph': 0}, {'kph': 100, 'known': 0.0}, {'kph': 100, 'ended': 1.0}])
def test_a_number_not_in_force_goes_nowhere(changes):
    assert trusting(net(), 3.0)(batch(2)[0], number_row(**changes))[0].abs().sum() == 0


def test_the_network_can_say_another_speed_than_the_number():
    from nnslr_tools.brain_targets import speed_class
    model = trusting(net(), 0.5, head=speed_class(120))
    assert model(batch(2)[0], number_row(60))[0].argmax(dim=-1).tolist() == [[speed_class(120)] * 2]
    assert trusting(net(), -1.0)(batch(2)[0], number_row(60))[0][0, 0, speed_class(60)] == -1


def test_the_trust_in_the_number_learns_from_a_frame_that_says_it():
    model = net()
    images, extra = batch(2)[0], number_row(60)
    speed, works, _state = model(images, extra)
    from nnslr_tools.brain_net import brain_loss
    from nnslr_tools.brain_targets import speed_class
    brain_loss(speed, works, torch.full((1, 2), speed_class(60)), torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)).backward()
    assert gradient_of(model, 'trust') > 0


def test_the_trust_in_the_number_is_one_value_that_comes_from_the_memory():
    from nnslr_tools.brain_targets import speed_class
    model = net()
    with torch.no_grad():
        model.speed_head.weight.zero_()
        model.speed_head.bias.zero_()
        said = model(batch(2)[0], number_row(60))[0][0, :, speed_class(60)]
    assert said[0] != said[1]
    assert model.trust.out_features == 1


def test_a_mark_does_not_reach_the_trust_through_the_number():
    model = net()
    backward(model, batch(2)[0], number_row(60), answers(2, works_on=[0, 1]))
    assert gradient_of(model, 'works_head') > 0
    assert gradient_of(model, 'trust') == 0


def test_with_no_signs_among_the_groups_there_is_no_number_to_add():
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import INPUT_SIZES
    groups = ('plate', 'hour', 'country')
    model = trusting(build_brain_net(groups, pretrained=False), 3.0)
    extra = torch.ones(1, 2, sum(INPUT_SIZES[name] for name in groups))
    assert model(batch(2)[0], extra)[0].abs().sum() == 0


def test_the_number_is_found_wherever_the_signs_are_among_the_groups():
    from nnslr_tools.brain_net import build_brain_net
    from nnslr_tools.brain_targets import speed_class
    groups = ('plate', 'hour', 'signs', 'country')
    model = trusting(build_brain_net(groups, pretrained=False), 3.0)
    assert model(batch(2)[0], number_row(80, groups=groups))[0].argmax(dim=-1).tolist() == [[speed_class(80)] * 2]


def test_the_car_gets_the_trust():
    from nnslr_tools.brain_net import car_weights
    assert 'trust.weight' in car_weights(net())

