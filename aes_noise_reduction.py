## example code for noise reduction

def noise_reduction(engine_context, state, n=16):
    engine = engine_context.get_engine()
    
    a = state
    
    if n == 256:
        p = 8
        
    if n == 16:
        p = 4
        
    for _ in range (p):
        if state.level == 0:
            state = engine.bootstrap(state, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
        state = engine.multiply(state, state, engine_context.get_relinearization_key())
        
    if state.level == 0:
        state = engine.bootstrap(state, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
        
    if a.level == 0:
        a = engine.bootstrap(a, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    state = engine.multiply(state, a, engine_context.get_relinearization_key())
        
    if state.level == 0:
        state = engine.bootstrap(state, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    state = engine.multiply(state, (-1/n))
        
    if a.level == 0:
        a = engine.bootstrap(a, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    a = engine.multiply(a, (1 + 1/n))
    
    out = engine.add(state, a)
    out = engine.bootstrap(out, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    out = engine.intt(out)
    
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
