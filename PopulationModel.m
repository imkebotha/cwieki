classdef PopulationModel < handle
    properties
        modelname;      % "ricker", "tl" (theta-logistic), "fa" (flexible-allee), "ml" (mate-limited)
        theta;          % parameters used to simulate the data
        theta_trans;    % transformed parameters
        T;              % number of observations
        y;              % observations
        y0;             % observation at time 0
        np;             % number of parameters
        param_support;  % support of each parameter

        names;          % names of the parameters
        tnames;         % names of the transformed parameters
        theta_block;    % indices of model parameters
        phi_block;      % indices of noise parameters                                   

        fixed_cov;      % likelihood covariance if fixed
        fixed_phi;      % noise parameters if fixed
        fixed_tphi      % transformed noise parameters if fixed
        
        % constants
        BETA0 = 1; BETA1 = 2; SIGMA;
        BETA = 3; % beta2, beta3 or beta4 depending on the model
    end
    
    methods
        %% constructor
        
        function o = PopulationModel(T, y0, modelname, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 3 || length(x) == 4 || isnan(x));
            addParameter(p, 'y', nan); 
            parse(p, varargin{:});
            
            % set values from input
            o.T = T;
            o.theta = p.Results.theta;
            o.y = p.Results.y;
            o.y0 = y0;
            o.modelname = modelname;
            
            % specify parameter blocks (only Ricker model is different)
            o.theta_block = [1 2 3];
            o.phi_block = 4;
            o.SIGMA = 4;
            o.np = 4;
            
            if lower(modelname) == "ricker"
                o.param_support = {[-inf inf];[-inf inf];[0 inf]};
                o.theta_block = [1 2];
                o.phi_block = 3;
                o.SIGMA = 3;
                
                % fixed values
                o.np = 3;
                o.tnames = {'BETA0';'BETA1';'LOG(SIGMA)'};
                o.names = {'$\beta_0$'; '$\beta_1$' ; '$\sigma$'}; 

            elseif lower(modelname) == "tl"
                o.param_support = {[-inf inf];[-inf inf];[-inf inf];[0 inf]};
                
                % fixed values
                o.tnames = {'BETA0';'BETA1';'BETA2';'LOG(SIGMA)'};
                o.names = {'$\beta_0$'; '$\beta_1$'; '$\beta_2$' ; '$\sigma$'};

            elseif lower(modelname) == "fa"
                o.param_support = {[-inf inf];[-inf inf];[-inf inf];[0 inf]};
                
                % fixed values
                o.tnames = {'BETA0';'BETA1';'BETA3';'LOG(SIGMA)'};
                o.names = {'$\beta_0$'; '$\beta_1$'; '$\beta_3$' ; '$\sigma$'};
            
            elseif lower(modelname) == "ml"
                o.param_support = {[-inf inf];[-inf inf];[0 inf];[0 inf]};
                
                % fixed values
                o.tnames = {'BETA0';'BETA1';'BETA4';'LOG(SIGMA)'};
                o.names = {'$\beta_0$'; '$\beta_1$'; '$\beta_4$' ; '$\sigma$'};
            end
            
            % simulate data if necessary
            o.theta_trans = o.transform(o.theta, false);
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    error("Need to specify a value for theta");
                end 
                o.y = [o.y0 zeros(1, o.T)];
                for t = 1:o.T
                    if o.modelname == "ricker" % Ricker
                        mu = o.theta(:, o.BETA0) + o.y(t) + o.theta(:, o.BETA1).*exp(o.y(t));
                    elseif o.modelname == "tl" % Theta-logistic
                        mu = o.theta(:, o.BETA0) + o.y(t) + o.theta(:, o.BETA1).*exp(o.theta(:, o.BETA)*o.y(t));
                    elseif o.modelname == "fa" % Flexible-allee
                        mu = o.theta(:, o.BETA0) + o.y(t) + o.theta(:, o.BETA1).*exp(o.y(t)) + o.theta(:, o.BETA)*exp(2*o.y(t));
                    elseif o.modelname == "ml" % Mate-limited
                        mu = o.theta(:, o.BETA0) + 2*o.y(t) + o.theta(:, o.BETA1).*exp(o.y(t)) - log(o.theta(:, o.BETA) + exp(o.y(t)));
                    end

                    o.y(t+1) = normrnd(mu, o.theta(o.SIGMA));
                end
                o.y = o.y(2:end); % remove y0         
            end
        end % PopulationModel(T, y0, modelname, varargin)

        %% Likelihood

        % fix the covariance of the likelihood
        function fix_noise(o, varargin)
            o.fixed_cov = [];
            o.fixed_phi = [];
            o.fixed_tphi = [];
            if ~isempty(varargin)
                params = varargin{:};
                tparams = o.transform(params, false);
                o.fixed_phi = params(o.phi_block);
                o.fixed_tphi = tparams(o.phi_block);
                o.fixed_cov = likelihood_cov(o, tparams);
            end
        end % fix_noise(o, varargin)


        % calculate the mean of the likelihood
        function mu = likelihood_mean(o, tparams)
            params = o.transform(tparams, true);
            ya = [o.y0 o.y(1:end-1)];

            if o.modelname == "ricker" % Ricker
                mu = params(:, o.BETA0) + ya + params(:, o.BETA1).*exp(ya);
            elseif o.modelname == "tl" % Theta-logistic
                mu = params(:, o.BETA0) + ya + params(:, o.BETA1).*exp(params(:, o.BETA)*ya);
            elseif o.modelname == "fa" % Flexible-allee
                mu = params(:, o.BETA0) + ya + params(:, o.BETA1).*exp(ya) + params(:, o.BETA)*exp(2*ya);
            elseif o.modelname == "ml" % Mate-limited
                mu = params(:, o.BETA0) + 2*ya + params(:, o.BETA1).*exp(ya) - log(params(:, o.BETA) + exp(ya));
            end
        end % likelihood_mean(o, tparams)

        
        % calculate the covariance of the likelihood
        function co = likelihood_cov(o, tparams)
            if isempty(o.fixed_cov)
                params = o.transform(tparams, true);
                co = ones(1, o.T)*params(o.SIGMA)^2;
                co = diag(co);
            else
                co = o.fixed_cov;
            end
        end % likelihood_cov(o, tparams)
        

        % calculate the likelihood (vectorised)
        function lp = loglikelihood(o, tparams)
            params = o.transform(tparams, true);
            
            if isempty(o.fixed_cov)
                sig = params(:, o.SIGMA);
            else
                sig = o.fixed_phi;
            end

            mu = likelihood_mean(o, tparams);
            lp = sum(norm_lpdf(o.y, mu, sig), 2);
        end % loglikelihood(o, tparams)

        
        %% Prior
        
        % calculate the log-pdf of the prior
        function lp = prior_lpdf(o, tparams)
            lp = norm_lpdf(tparams(:, o.BETA0), 1, 1);
            lp = lp + norm_lpdf(tparams(:, o.BETA1), 0, 1);
            
            if o.modelname == "tl"
                lp = lp + norm_lpdf(tparams(:, o.BETA), 1, 0.2);
            elseif o.modelname == "fa"
                lp = lp + norm_lpdf(tparams(:, o.BETA), 0, 1);
            elseif o.modelname == "ml"
                lp = lp + logHN_lpdf(tparams(:, o.BETA), 1);
            end
            
            if isempty(o.fixed_cov)
                lp = lp + logHN_lpdf(tparams(:, o.SIGMA), 1);
            end
        end % prior_lpdf(o, tparams)
        

        % calculate the log-pdf of the prior for parameter i
        function lp = priori_lpdf(o, params, i)
            if i == o.BETA0
                lp = norm_lpdf(params, 1, 1);
            elseif i == o.BETA1
                lp = norm_lpdf(params, 0, 1);
            elseif i == o.BETA
                if o.modelname == "tl"
                    lp = norm_lpdf(params, 1, 0.2);
                elseif o.modelname == "fa"
                    lp = norm_lpdf(params, 0, 1);
                else
                    lp = logHN_lpdf(log(params), 1);
                end
            else
                lp = logHN_lpdf(log(params), 1);
            end

        end % priori_lpdf(o, params, i)
        

        % simulate from the prior
        function tparams = prior_rnd(o, N)
            if isempty(o.fixed_cov)
                tparams = zeros(N, o.np);
            else
                tparams = zeros(N, o.np-1);
            end
            tparams(:, o.BETA0) = normrnd(1, 1, N, 1);
            tparams(:, o.BETA1) = normrnd(0, 1, N, 1);
            
            if o.modelname == "fa"
                tparams(:, o.BETA) = normrnd(0, 1, N, 1);
            elseif o.modelname == "tl"
                tparams(:, o.BETA) = normrnd(1, 0.2, N, 1);
            elseif o.modelname == "ml"
                tparams(:, o.BETA) = log(abs(normrnd(0, 1, N, 1)));
            end
            
            if isempty(o.fixed_cov)
                tparams(:, o.SIGMA) = log(abs(normrnd(0, 1, N, 1)));
            end
        end % prior_rnd(o, N)
        
        
        %% Other
        
        % transform the parameters, can be the inverse transformation
        function uparams = transform(o, uparams, inverse)
            if o.modelname == "ml"
                if inverse
                    uparams(:, o.BETA) = exp(uparams(:, o.BETA));
                else
                    uparams(:, o.BETA) = log(uparams(:, o.BETA));
                end
            end
            
            if isempty(o.fixed_cov)
                if inverse
                    uparams(:, o.SIGMA) = exp(uparams(:, o.SIGMA)); 
                else
                    uparams(:, o.SIGMA) = log(uparams(:, o.SIGMA));
                end
            end
        end % transform(o, theta, inverse)

    end % methods
    
end % PopulationModel

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log-transformed half-normal distribution
function y = logHN_lpdf(logx, s)
    y = 0.5*log(2/pi)-log(s)-exp(2*logx)/(2*s^2) + logx;
end

