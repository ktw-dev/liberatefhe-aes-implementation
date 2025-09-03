## example code for noise reduction

from engine_context import CKKS_EngineContext

def noise_reduction(engine_context: CKKS_EngineContext, state, n=16):    
    a = state
    
    if n == 256:
        p = 8
        
    if n == 16:
        p = 4
        
    for _ in range (p):
        if state.level == 0:
            state = engine_context.ckks_bootstrap(state)
        state = engine_context.ckks_multiply(state, state)
        
    if state.level == 0:
        state = engine_context.ckks_bootstrap(state)
        
    if a.level == 0:
        a = engine_context.ckks_bootstrap(a)
    state = engine_context.ckks_multiply(state, a)
        
    if state.level == 0:
        state = engine_context.ckks_bootstrap(state)
    state = engine_context.ckks_multiply(state, (-1/n))
        
    if a.level == 0:
        a = engine_context.ckks_bootstrap(a)
    a = engine_context.ckks_multiply(a, (1 + 1/n))
    
    out = engine_context.ckks_add(state, a)
    out = engine_context.ckks_bootstrap(out)
    
    return out

if __name__ == "__main__":
    ## how to use the noise reduction function

    engine = "your_he_engine_here"
    hi_state = "your_high_noise_state_here"
    lo_state = "your_low_noise_state_here"
    state = "your_initial_state_here"

    hi_state = noise_reduction(engine, hi_state , 16)
    lo_state = noise_reduction(engine, lo_state , 16)

    state = noise_reduction(engine, state, 256) # 8bits
