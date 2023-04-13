function results = CWIEKI(m, N, E, M)

% initialise
acc_rate_N = [];
gamma_t = 0; 
temp_hist = 0;
part_w = ones(N,1)/N;
Gamma = cell(N, 1);
loglike = zeros(N, 1);

% sample from the prior and calculate the likelihood
theta = m.prior_rnd(N);
G = m.likelihood_mean(theta);
for i = 1:N
    Gamma{i} = m.likelihood_cov(theta(i, :));
    loglike(i) = logmvnpdf(m.y, G(i, :), Gamma{i});
end

while gamma_t < 1 
    %% determine gamma_{t+1}

    ess = calc_ess(1, gamma_t, part_w, loglike);
    if ess > E 
        newgamma = 1;
    else
        newgamma = fzero(@(newgamma) calc_ess(newgamma, gamma_t, part_w, loglike) - E, [gamma_t + eps 1]);
    end
    fprintf('next temperature is %f\n',newgamma);
    temp_hist = [temp_hist, newgamma];
    hn_inv = 1/(newgamma - gamma_t);
    gamma_t = newgamma;
    
    %% update model parameters
    
    Cgg = cov(G); 
    Cgtheta = cross_cov(theta(:, m.theta_block), G); 
    for i = 1:N 
        % IEKI update
        theta(i, m.theta_block) = theta(i, m.theta_block) + (Cgtheta*pinv(Cgg + hn_inv*Gamma{i})*(m.y - G(i, :) - mvnrnd(zeros(1, m.T), hn_inv*Gamma{i}))')';
        
        % update likelihood
        G(i, :) = m.likelihood_mean(theta(i, :));
        loglike(i) = logmvnpdf(m.y, G(i, :), Gamma{i});
    end
    
    %% update noise parameters
    
    % initialise
    cov_rw_noise = cov(theta(:, m.phi_block));
    acc_rate = zeros(N, 1);

    % current
    prior_curr = m.prior_lpdf(theta); 
    log_post_curr = gamma_t*loglike + prior_curr;

    % perform M MCMC iterations
    for k = 1:M
        for i = 1:N
            % proposal
            theta_prop = theta(i, :);
            theta_prop(m.phi_block) = mvnrnd(theta(i, m.phi_block), cov_rw_noise); 
            
            Gamma_prop = m.likelihood_cov(theta_prop);
            loglike_prop = logmvnpdf(m.y, G(i, :), Gamma_prop);
            prior_prop = m.prior_lpdf(theta_prop);
            log_post_prop = gamma_t*loglike_prop + prior_prop; 

            % accept/reject
            mh = exp(log_post_prop - log_post_curr(i));
            if mh > rand()
                theta(i, :) = theta_prop;
                Gamma{i} = Gamma_prop;
                loglike(i) = loglike_prop;
                log_post_curr(i) = log_post_prop;
                acc_rate(i) = acc_rate(i) + 1;
            end
        end
    end
    acc_rate = acc_rate./M;
    acc_rate_N = [acc_rate_N sum(acc_rate > 0)/N];
    
end

% update results
theta = m.transform(theta, true);
results.samples = theta;
results.mean = mean(theta);
results.temp_hist = temp_hist;
results.acc_rate_N = acc_rate_N;
results.penalty = (length(temp_hist) - 1) * N;

end

