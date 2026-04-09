from attacks.norm_attack import norm_attack
from attacks.direction_attack import direction_attack
from attacks.model_completion import model_completion_attack
from attacks.embedding_extension import embedding_extension_attack

ATTACK_REGISTRY = {
    "norm": norm_attack,
    "direction": direction_attack,
    "model_completion": model_completion_attack,
    "embedding_extension": embedding_extension_attack,
}
