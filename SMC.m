function results = SMC(m, N, E, c)

% initialise
S = 5; % initial number of MCMC trial iterations
R_hist = [];
part_w = ones(N,1)/N;
gamma_t = 0;
temp_hist = gamma_t;

% sample from the prior and calculate the likelihood
theta = m.prior_rnd(N);
loglike = m.loglikelihood(theta);

while gamma_t < 1
    %% determine gamma_{t+1}
    ess1 = calc_ess(1, gamma_t, part_w, loglike);
    if ess1 > E
        newgamma = 1;
    else
        newgamma = fzero(@(newgamma) calc_ess(newgamma,gamma_t,part_w,loglike)-E,[gamma_t+eps 1]);
    end
    fprintf('next temperature is %f\n',newgamma);
    temp_hist = [temp_hist, newgamma];
    
    %% reweight

    logW = log(part_w) + (newgamma - gamma_t)*loglike;
    logNW = logW - logsumexp(logW);
    part_w = exp(logNW);
    
    % compute effective sample size
    ess = 1/sum(part_w.^2);
    fprintf('ESS is %f\n',ess);

    gamma_t = newgamma;
    
    %% resample-move
    fprintf('**** Doing a resample-move ******\n');
    
    % resampling multinomially
    ind = randsample(1:N, N, true, part_w);
    theta = theta(ind, :);
    loglike = loglike(ind);
    part_w = ones(N,1)/N;
    
    % covariance for MCMC
    cov_rw = 2.38^2/m.np*cov(theta);
    
    % perform S MCMC iterations
    count = zeros(N,1);
    for i = 1:N
        for k = 1:S
            % proposal
            prop = mvnrnd(theta(i, :), cov_rw);
 
            %compute loglikelihood at proposal
            part_loglike_prop = m.loglikelihood(prop);
            
            log_prior_curr = m.prior_lpdf(theta(i,:));
            log_prior_prop = m.prior_lpdf(prop);
            
            MHR = exp(gamma_t*(part_loglike_prop - loglike(i))...
                + log_prior_prop - log_prior_curr);
            if (MHR > rand)
                %then accept proposal
                theta(i, :) = prop;
                loglike(i) = part_loglike_prop;
                count(i) = count(i) + 1;
            end
        end
    end
    
    % estimate MCMC acceptance probability
    p = sum(count)/(S*N);
    
    R = ceil(log(c)/log(1-p));
    R_hist = [R_hist ; R];
    disp(['Adapted R: ',num2str(R)])
    
	% run the remaining R-S MCMC iterations
    for i = 1:N
        for k = 1:(R-S)
            % proposal
            prop = mvnrnd(theta(i, :), cov_rw);
            
            part_loglike_prop = m.loglikelihood(prop);
            
            log_prior_curr = m.prior_lpdf(theta(i, :));
            log_prior_prop = m.prior_lpdf(prop);
            
            MHR = exp(gamma_t*(part_loglike_prop - loglike(i))...
                + log_prior_prop - log_prior_curr);
            if (MHR > rand)
                %then accept proposal
                theta(i, :) = prop;
                loglike(i) = part_loglike_prop;
            end
        end
    end
    fprintf('the number of unique particles after resample-move is %d\n',...
        length(unique(theta(:, 1))));
    
    S = floor(0.5*R); % update number of trial mcmc iterations to use next 
end

% update results
theta = m.transform(theta, true);
results.samples = theta;
results.temp_hist = temp_hist;
results.R_hist = R_hist;
results.penalty = N * sum(R_hist);
results.mean = mean(theta);

end