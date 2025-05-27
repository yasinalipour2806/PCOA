def PCOA(N, T, fitness, lb, ub, dim):
    def gaussian_chaotic_value():
        z = np.random.normal(0, 1)
        g = 1.0 / (1.0 + np.exp(-z))
        return g  # مقداری در (0,1)

    def gaussian_quasi_reflection(X, fit, Bast_P, fbest, lb, ub, fitness,t,T):
        mid = 0.5 * (lb + ub)  # (lb+ub)/2
        for i in range(X.shape[0]):
            x = X[i, :].copy()

            # 1) Opposite
            x_op = lb + ub - x * np.exp(-t/T)*((-1)**(t+1))

            # 2) Quasi-Opposite (x_qo): در بازه [ (lb+ub)/2, x_op ]
            lower_bound = np.minimum(mid, x_op)
            upper_bound = np.maximum(mid, x_op)
            x_qo = lower_bound + np.random.rand(len(x)) * (upper_bound - lower_bound)
            x_qo = np.clip(x_qo, lb, ub)

            # 3) Gaussian-Chaos factor
            gamma_val = gaussian_chaotic_value()

            # 4) Quasi-Reflection (x_gqo):
            #    x_gqo = rand( mid, lb+ub - gamma*x )
            reflect_center = lb + ub - gamma_val * x
            reflect_lower = np.minimum(mid, reflect_center)
            reflect_upper = np.maximum(mid, reflect_center)
            x_gqo = reflect_lower + np.random.rand(len(x)) * (reflect_upper - reflect_lower)
            x_gqo = np.clip(x_gqo, lb, ub)

            # 5) Assessment
            f_qo = fitness(x_qo)
            f_gqo = fitness(x_gqo)

            if f_qo < fit[i] or f_gqo < fit[i]:
                if f_qo < f_gqo:
                    X[i, :] = x_qo
                    fit[i] = f_qo
                else:
                    X[i, :] = x_gqo
                    fit[i] = f_gqo

                if fit[i] < fbest:
                    Bast_P = X[i, :].copy()
                    fbest = fit[i]

        return X, fit, Bast_P, fbest

    def local_refinement(Best_solution, best_score, fitness, lb, ub, dim, t, T):
        radius = (ub[0]-lb[0]) * 0.1 * (1 - t/T)
        if radius <= 1e-100:
            return Best_solution, best_score

        num_neighbors = 10

        improved = False
        new_best = Best_solution.copy()
        new_best_score = best_score

        for _ in range(num_neighbors):
            direction = np.random.randn(dim)
            direction = direction / (np.linalg.norm(direction) + 1e-30)
            candidate = Best_solution + direction * radius * (np.random.rand() * 1.0)
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = fitness(candidate)
            if candidate_fitness < new_best_score:
                new_best = candidate.copy()
                new_best_score = candidate_fitness
                improved = True
        
        return new_best, new_best_score


    def levy_flight(dim, beta=1.5):
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / np.abs(v)**(1 / beta)
        return step * 0.01  
 
    def calculate_beta(t, T, beta_min=1.5, beta_max=2.0, steepness=4.1, p=1.5):
        """
        Calculate a smoothly increasing beta value with accelerating growth.

        Parameters:
            t (int): Current iteration.
            T (int): Total iterations.
            beta_min (float): Minimum beta value (default: 1.5).
            beta_max (float): Maximum beta value (default: 2.0).
            steepness (float): Controls the steepness of the curve (default: 5).
            p (float): Controls how growth accelerates (default: 2).

        Returns:
            float: Calculated beta value.
        """
        progress = t / T
        beta = beta_min + (beta_max - beta_min) * (1 - np.exp(-steepness * progress**p)) 
        return beta
        
    def generate_offspring_with_limited_levy(parent, num_offspring, r, fitness, lb, ub, dim, beta=1.8):
        """
        Generate offspring using Levy flight limited to radius r around the parent.
        """
        offspring = np.zeros((num_offspring, dim))
        for i in range(num_offspring):
            # Generate Levy step
            RL = levy_flight(dim, beta)
            
            # Scale Levy step to fit within radius r
            if np.linalg.norm(RL) == 0:
                step = RL
            else:
                step = RL * r / np.linalg.norm(RL)
            offspring[i, :] = parent + step

            # Ensure offspring remain within bounds
            offspring[i, :] = np.clip(offspring[i, :], lb, ub)
        
        # Calculate fitness for offspring
        offspring_fitness = np.array([fitness(offspring[i, :]) for i in range(num_offspring)])
        return offspring, offspring_fitness

    def apply_elitism_with_rollback(X, fit, Bast_P, fbest, lb, ub, fitness, elite_rate=0.1):
        """
        Apply elitism with rollback if it does not improve the result.
        """
        # Save current state
        X_backup = X.copy()
        fit_backup = fit.copy()
        Bast_P_backup = Bast_P.copy()
        fbest_backup = fbest

        # Number of elites to keep
        num_elites = max(1, int(elite_rate * X.shape[0]))
        elite_indices = np.argsort(fit)[:num_elites]  # Get indices of best agents
        worst_indices = np.argsort(fit)[-num_elites:]  # Get indices of worst agents

        # Replace worst agents with best agents
        for i in range(num_elites):
            X[worst_indices[i], :] = X[elite_indices[i], :]

        # Re-evaluate fitness
        fit = np.array([fitness(X[i, :]) for i in range(X.shape[0])])
        best, location = np.min(fit), np.argmin(fit)

        # Update global best if improved
        if best < fbest:
            Bast_P = X[location, :].copy()
            fbest = best
        else:
            # Rollback to previous state
            X = X_backup
            fit = fit_backup
            Bast_P = Bast_P_backup
            fbest = fbest_backup

        return X, fit, Bast_P, fbest

    def DE_crossover(target, mutant, crossover_rate=0.1):
        """
        Perform binomial crossover between target and mutant vectors.
        
        Parameters:
            target (np.ndarray): The target vector (current agent).
            mutant (np.ndarray): The mutant vector generated by mutation.
            crossover_rate (float): Probability of crossover for each dimension.
        
        Returns:
            np.ndarray: The trial vector after crossover.
        """
        dim = len(target)
        trial = np.copy(target)
        j_rand = np.random.randint(0, dim)
        
        for j in range(dim):
            if np.random.rand() < crossover_rate or j == j_rand:
                trial[j] = mutant[j]
        
        return trial

    # For plotting and tracking
    Convergence_curve = np.zeros(T)
    history = np.zeros((N, dim, T))        
    trajectory = np.zeros((T, dim))           
    Average_fitness = np.zeros(T) 

    lb = np.ones(dim) * lb
    ub = np.ones(dim) * ub

    # INITIALIZATION
    X = np.zeros((N, dim))
    for i in range(dim):
        X[:, i] = lb[i] + np.random.rand(N) * (ub[i] - lb[i])

    fit = np.zeros(N)
    for i in range(N):
        M = X[i, :]
        fit[i] = fitness(M)

    best_temp = np.zeros((T, dim))
    # MAIN LOOP
    for t in range(1, T + 1):
        CF = 1 - ((t / T)**3 * (10 - 15 * (t / T) + 6 * (t / T)**2))   # Advanced Smoothstep Function
        LO_LOCAL = lb / (t + 1) 
        HI_LOCAL = ub / (t + 1) 
        best, location = np.min(fit), np.argmin(fit)
        if t == 1:
            Bast_P = X[location, :].copy()
            fbest = best
        elif best < fbest:
            fbest = best
            Bast_P = X[location, :].copy()
        best_temp[t-1, :] = Bast_P
        beta = calculate_beta(t, T)

        for i in range(N):
            if t < 1 * T / 3:  # Search  stage
                Rn = X.shape[0]
                X_random_1 = X[np.random.randint(0, Rn), :]
                X_random_2 = X[np.random.randint(0, Rn), :]
                R1 = np.random.rand(dim)
                X1 = X[i, :] + (X_random_1 - X_random_2) * R1
            elif 1 * T / 3 <= t < 2 * T / 3:  # Approaching  stage

                RB = np.clip(np.random.rand(dim), LO_LOCAL, HI_LOCAL)
                qw = np.exp(-5 * (t/T)**2)
                X1 = Bast_P + qw * (RB - 0.5) * (Bast_P - X[i, :])
                
            else:   # any name as compatible with the movie
                RL = levy_flight(dim, beta)
                X1 = Bast_P + CF * X[i, :] * RL
                # Boundary check
                X1 = np.clip(X1, lb, ub)
                # Evaluate the new position
                f_newP1 = fitness(X1)
                if f_newP1 <= fit[i]:
                    X[i, :] = X1
                    fit[i] = f_newP1
                    num_offspring = np.random.randint(10, 21)
                    r = 0.1 * (ub[0] - lb[0]) * (1 - t / T)
                    offspring, offspring_fitness = generate_offspring_with_limited_levy(
                        X[i, :], num_offspring, r, fitness, lb, ub, dim, beta
                    )
                    best_offspring_idx = np.argmin(offspring_fitness)
                    if offspring_fitness[best_offspring_idx] < fit[i]:
                        X[i, :] = offspring[best_offspring_idx]
                        fit[i] = offspring_fitness[best_offspring_idx]
                    continue
                else:
                    RL = np.random.rand(dim) * levy_flight(dim, beta)
                    X1 = Bast_P + CF * X[i, :] * RL

            X1 = np.clip(X1, lb, ub)

            # Evaluate the new position
            f_newP1 = fitness(X1)
            if f_newP1 <= fit[i]:
                X[i, :] = X1
                fit[i] = f_newP1

        alpha_initial = 1
        alpha_mid = 0.5
        eta = 1
        p = 2
        beta_perturb = 5
        gammas = 50
        delta = 0.1
        for i in range(N):
            k = np.random.randint(0, N)
            Xrandom = X[k, :]
            
            r = np.random.rand()
            if r < 0.5:  
                RB = np.random.rand(dim)
                pf = 1 / (1 + eta * (t / T) ** p)
                exp_initial = alpha_initial * np.exp(-beta_perturb * (t / T))
                exp_mid = alpha_mid * np.exp(-gammas * ((t / T) - 0.5) ** 2)
                sinusoidal = 1 + delta * np.sin(2 * np.pi * (t / T))
                Perturbation = ((exp_initial + exp_mid) * sinusoidal) * pf
                # X2 = Bast_P + Perturbation * (2 * RB - 1) * X[i, :]
                # X2 = exp_initial = alpha_initial * np.exp(-beta_perturb * (t / T))
                X2=0
                if np.random.rand()>=0.5:
                    X2 = sinusoidal = 1 + delta * np.sin(2 * np.pi * (t / T))
                    # Boundary check
                    X2 = np.clip(X2, lb, ub)
        
                    # Evaluate the new position
                    f_newP2 = fitness(X2)
                    if f_newP2 <= fit[i]:
                        X[i, :] = X2
                        fit[i] = f_newP2
                    else:
                        X2 = Bast_P +  Perturbation * (2 * RB - 1) * X[i, :]
                # X2 = Bast_P + (2 * RB - 1) * X[i, :]
            else: 
                R2 = np.random.rand(dim)
                K = np.round(1 + np.random.rand())
                # X2 = X[i, :] + R2 * (Xrandom - K * X[i, :])
                X2 = exp_initial = alpha_initial * np.exp(-beta_perturb * (t / T))
                # Boundary check
                X2 = np.clip(X2, lb, ub)

                # Evaluate the new position
                f_newP2 = fitness(X2)
                if (f_newP2 <= fit[i]):
                    X[i, :] = X2
                    fit[i] = f_newP2
                    # continue
                else:
                    R2 = np.random.rand(dim)
                    l=np.random.uniform(-1,1)
                    #X2 = X[i, :] + (1 - 2 * np.random.rand()) * (LO_LOCAL + R2 * (HI_LOCAL - LO_LOCAL))
                    # X2 =   X[i, :] + np.random.rand() * np.cos(0.5*np.pi * t/T) * np.abs(( f_newP2**2 - X[i, :])**2)
                    X2 = (np.abs(f_newP2**2 - X[i, :]**2)) * np.exp(5*l) * np.cos(2*np.pi* l) + f_newP2
            # Boundary check
            X2 = np.clip(X2, lb, ub)

            # Evaluate the new position
            f_newP2 = fitness(X2)
            if f_newP2 <= fit[i]:
                X[i, :] = X2
                fit[i] = f_newP2
        X, fit, Bast_P, fbest = gaussian_quasi_reflection(X, fit, Bast_P, fbest, lb, ub, fitness,t,T)

        history[:, :, t-1] = X[:, :]
        trajectory[t-1, :] = X[0, :]
        Average_fitness[t-1] = np.mean(fit)
        Convergence_curve[t - 1] = fbest

        # Apply Elitism with Rollback
        X, fit, Bast_P, fbest = apply_elitism_with_rollback(X, fit, Bast_P, fbest, lb, ub, fitness, elite_rate=0.1)

        # Add DE-style Crossover and Rollback
        for i in range(N):
            idxs = list(range(N))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            mutant = X[a, :] + 0.5 * (X[b, :] - X[a, :])  

            trial = DE_crossover(X[i, :], mutant, crossover_rate=0.1)

            trial = np.clip(trial, lb, ub)

            trial_fitness = fitness(trial)

            if trial_fitness < fit[i]:
                backup_X = X[i, :].copy()
                backup_fit = fit[i]

                X[i, :] = trial
                fit[i] = trial_fitness

                if trial_fitness < fbest:
                    Bast_P = trial.copy()
                    fbest = trial_fitness
        Bast_P, fbest = local_refinement(Bast_P, fbest, fitness, lb, ub, dim, t, T)

    return Bast_P, fbest, Convergence_curve, history, trajectory, Average_fitness


for algo_D in ['PCOA'] :
    statistics_all_benchmarks[algo_D] = []
    for fn in range(1,11):        
        cec2019 = get_functions_by_classname(f"F{fn}2019")
        cec = cec2019[0]()
        dim = cec.dim_default
        
        best_posList = []
        best_scoreLsit = []
        curveList = []
        trajectoryList = []
        avg_fitnessList = []
        search_historyList = []

        results = Parallel(n_jobs=-1)(delayed(PCOA )(SearchAgents, Max_iter, cec.evaluate,
                                                   cec.lb[0], cec.ub[0], dim) for _ in range(num_runs))
        
        for i, (best_pos, best_score, Convergence_curve, history, trajectory, Average_fitness) in enumerate(results):
            trajectoryList.append(trajectory)
            avg_fitnessList.append(Average_fitness)
            best_posList.append(best_pos)
            best_scoreLsit.append(best_score)
            curveList.append(Convergence_curve)
            search_historyList.append(history)        
    
        indx = best_scoreLsit.index(min(best_scoreLsit))

        statistics_all_benchmarks[algo_D].append({
            'Fn' : f'F{fn}',
            "Mean": f'{np.mean(best_scoreLsit):.3E}',
            "Best": f'{np.min(best_scoreLsit):.3E}',
            "Worst": f'{np.max(best_scoreLsit):.3E}',
            "Median": f'{np.median(best_scoreLsit):.3E}' ,
            "STD": f'{np.std(best_scoreLsit):.3E}',
            'Best_position': best_posList[indx],
            'Curve': curveList[indx],
            'Best_result': best_scoreLsit,
            "Trajectory_curve": trajectoryList[indx],
            "Avg_fitness": avg_fitnessList[indx],
            'Search_history': search_historyList[indx],
        })

        print(dict(list(statistics_all_benchmarks[algo_D][-1].items())[:6]))
