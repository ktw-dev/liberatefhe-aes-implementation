## example code for noise reduction

def noise_reduction_poly(engine_context, x, n):
    print("Noise Reduction Start !")
    engine = engine_context.get_engine()
    a = x
    if n == 256:
        p = 8
        
    if n == 16:
        p = 4
        
    for _ in range (p):
        if x.level == 0:
            x = engine.bootstrap(x, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
        x = engine.multiply(x, x, engine_context.get_relinearization_key())
        
    if x.level == 0:
        x = engine.bootstrap(x, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    if a.level == 0:
        a = engine.bootstrap(a, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    x = engine.multiply(x, a, engine_context.get_relinearization_key())
        
    if x.level == 0:
        x = engine.bootstrap(x, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    x = engine.multiply(x, (-1/n))
        
    if a.level == 0:
        a = engine.bootstrap(a, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    a = engine.multiply(a, (1 + 1/n))
    
    print("Noise Reduction Success !")
    out = engine.add(x, a)
    out = engine.bootstrap(out, engine_context.get_relinearization_key(), engine_context.get_conjugation_key(), engine_context.get_bootstrap_key())
    
    return out

## how to use the noise reduction function

engine = "your_he_engine_here"
hi_state = "your_high_noise_state_here"
lo_state = "your_low_noise_state_here"
state = "your_initial_state_here"

hi_state = noise_reduction_poly(engine, hi_state , 16) # 4bits
lo_state = noise_reduction_poly(engine, lo_state , 16)

state = noise_reduction_poly(engine, state, 256) # 8bits
