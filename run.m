%% Common Parameters

seed = 1;
N = 1000;   % # particles
E = N*0.5;  % ESS resampling threshold
M = 1000;   % # MCMC iterations for updating noise parameters
c = 0.01;   % probability all particles move at least once during MCMC is 1-c

%% Linear Gaussian Model

% create model
rng(seed)

T = 50; b = 0.5; sig = 0.5;
theta = [b sig];
m = LinearGaussianModel(T, 'theta', theta);

% SMC
rng(seed)
results_smc = SMC(m, N, E, c);

% CW-IEKI
rng(seed)
results_cwieki = CWIEKI(m, N, E, M);

% IEKI (with fixed noise)
thetas = repmat(theta, 3, 1);
thetas(:, end) = sig.*[1; 0.5; 2];
results_ieki = cell(3, 1);
for i = 1:3 
    rng(seed)
    m.fix_noise(thetas(i, :))
    results_ieki{i} = IEKI(m, N, E);
end
m.fix_noise();

% save results
save("results_LG.mat", 'm', 'results_smc', 'results_cwieki', 'results_ieki');
gen_figures(m, results_smc, results_cwieki, results_ieki, "LG");


%% Population Models

modelnames = ["ricker"; "tl"; "fa"; "ml"]; 
b0 = 2; b1 = -0.5; b2 = 1; b3 = 0; b4 = 0; sig = 1; y0 = 2;
T = 50;

for j = 1:4
    % create model
    switch modelnames(j)
        case "ricker"
            theta = [b0 b1 sig];
        case "tl" 
            theta = [b0 b1 b2 sig];    
        case "fa"
            theta = [b0 b1 b3 sig];
        case "ml"
            theta = [b0 b1 b4 sig];
        otherwise
            error("Unknown model");
    end
    rng(seed)
    m = PopulationModel(T, y0, modelnames(j), 'theta', theta);
    figure; plot(1:m.T, exp(m.y))

    % SMC
    rng(seed)
    results_smc = SMC(m, N, E, c);
    
    % CW-IEKI
    rng(seed)
    results_cwieki = CWIEKI(m, N, E, M);
    
    % IEKI (with fixed noise)
    thetas = repmat(theta, 3, 1);
    thetas(:, end) = sig.*[1; 0.5; 2];
    results_ieki = cell(3, 1);
    for i = 1:3 
        rng(seed)
        m.fix_noise(thetas(i, :))
        results_ieki{i} = IEKI(m, N, E);
    end
    m.fix_noise();
    
    % save results
    save("results_" + m.modelname + ".mat", 'm', 'results_smc', 'results_cwieki', 'results_ieki');
    gen_figures(m, results_smc, results_cwieki, results_ieki, m.modelname);
end




