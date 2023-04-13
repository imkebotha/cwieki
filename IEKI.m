function results = IEKI(m, N, E)

% initialise
gamma_t = 0;
temp_hist = gamma_t;
loglike = zeros(N, 1);
part_w = ones(N,1)/N;
 
% sample from the prior and calculate the likelihood
Gamma = m.fixed_cov;
theta = m.prior_rnd(N); 
G = m.likelihood_mean(theta);
for i = 1:N
    loglike(i) = logmvnpdf(m.y, G(i, :), Gamma);
end

while gamma_t < 1
    %% determine gamma_{t+1} 

    ess = calc_ess(1, gamma_t, part_w, loglike);
    if ess > E
        newgamma = 1;
    else
        newgamma = fzero(@(newgamma) calc_ess(newgamma, gamma_t, part_w, loglike)-E,[gamma_t+eps 1]);
    end
    fprintf('next temperature is %f\n',newgamma);
    temp_hist = [temp_hist, newgamma];
    hn_inv = 1/(newgamma - gamma_t);
    gamma_t = newgamma;

    %% update model parameters

    Cgg = cov(G); 
    Cgtheta = cross_cov(theta, G); 
    K = Cgtheta*pinv(Cgg + hn_inv*Gamma);
    for i = 1:N
        theta(i, :) = theta(i, :) + (K*(m.y - G(i, :) - mvnrnd(zeros(1, m.T), hn_inv*Gamma))')';   

        % update likelihood
        G(i, :) = m.likelihood_mean(theta(i, :));
        loglike(i) = logmvnpdf(m.y, G(i, :), Gamma);
    end
end

% update results
theta = m.transform(theta, true);
results.samples = theta;
results.mean = mean(theta);
results.temp_hist = temp_hist;
results.phi = m.fixed_phi;
results.penalty = (length(temp_hist) - 1) * N;

end

