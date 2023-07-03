import sys
sys.path.insert(0, 'karel_env/dsl')
from dsl_prob_option_v2 import DSLProb_option_v2


def get_DSL_option_v2(dsl_type='prob', seed=None, environment='karel'):
    if dsl_type == 'prob':
        return DSLProb_option_v2(seed=seed, environment=environment)
    else:
        raise ValueError('Undefined dsl type')


