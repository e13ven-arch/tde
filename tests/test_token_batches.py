import random
from tde.train import TrainConfig, _token_batches


def test_token_batches_respect_budget_and_cover_all():
    lengths = [random.Random(i).randint(50, 4000) for i in range(3000)]
    cfg = TrainConfig(batch_size=32, max_tokens=24000)
    batches = _token_batches(list(range(3000)), lengths, cfg, random.Random(0))
    seen = sorted(j for b in batches for j in b)
    assert seen == list(range(3000))
    for b in batches:
        assert len(b) <= 32
        assert len(b) * max(lengths[j] for j in b) <= 24000
    # long rows end up in small batches, short rows in full ones
    assert min(len(b) for b in batches) < 32 < max(len(b) for b in batches) + 1
