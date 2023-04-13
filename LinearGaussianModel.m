classdef LinearGaussianModel < handle
    properties
        theta;          % parameters used to simulate the data
        theta_trans;    % transformed parameters
        T;              % number of observations
        y;              % observations
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
        B = 1; SIGMA = 2; 
    end
    
    methods
        %% constructor

        function o = LinearGaussianModel(T, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 2 || isnan(x));
            addParameter(p, 'y', nan); 
            parse(p, varargin{:});
            
            % fixed values
            o.theta_block = 1;
            o.phi_block = 2;
            o.np = 2;
            o.param_support = {[-inf inf] ; [0 inf]};
            o.tnames = {'B';'LOG(SIGMA)'};
            o.names = {'$b$'; '$\sigma$'};
                
            % set values from input
            o.T = T;
            o.theta = p.Results.theta;
            o.y = p.Results.y;
            o.fixed_cov = [];
            o.fixed_phi = [];
            o.fixed_tphi = [];
            
            % simulate data if necessary
            o.theta_trans = o.transform(o.theta, false);
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    error("Need to specify a value for theta");
                end 
                mu = likelihood_mean(o, o.theta_trans);
                Sig = likelihood_cov(o, o.theta_trans);
                o.y = mvnrnd(mu, Sig);      
            end
        end % LinearGaussianModel(T, varargin)

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
            mu = params(:, o.B).*ones(1, o.T);
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

            if isempty(o.fixed_phi)
                sig = params(:, o.SIGMA);
            else
                sig = o.fixed_phi;
            end

            lp = sum(norm_lpdf(o.y, params(:, o.B).*ones(1, o.T), sig), 2);
        end % loglikelihood(o, tparams)

        
        %% Prior
        
        % calculate the log-pdf of the prior
        function lp = prior_lpdf(o, tparams)
            if isempty(o.fixed_cov)
                lp = norm_lpdf(tparams(:, o.B), 0.5, 1) ...
                    + norm_lpdf(tparams(:, o.SIGMA), 1, 1);
            else
                lp = norm_lpdf(tparams(:, o.B), 0.5, 1);
            end
        end % prior_lpdf(o, tparams)
        

        % calculate the log-pdf of the prior for parameter i
        function lp = priori_lpdf(o, tparams, i)
            if i == o.B
                lp = norm_lpdf(tparams, 0.5, 1);
            else
                lp = norm_lpdf(tparams, 1, 1);
            end
        end % priori_lpdf(o, tparams, i)
        

        % simulate from the prior
        function tparams = prior_rnd(o, N)
            if isempty(o.fixed_cov)
                tparams = zeros(N, o.np);
                tparams(:, o.B) = normrnd(0.5, 1, N, 1);
                tparams(:, o.SIGMA) = normrnd(1, 1, N, 1);
            else
                tparams = normrnd(0.5, 1, N, 1);
            end
        end % prior_rnd(o, N)
        
        
        %% Other
        
        % transform the parameters, can be the inverse transformation
        function uparams = transform(o, uparams, inverse)
            if isempty(o.fixed_cov)
                if inverse
                    uparams(:, o.SIGMA) = exp(uparams(:, o.SIGMA)); 
                else
                    uparams(:, o.SIGMA) = log(uparams(:, o.SIGMA)); 
                end
            end
        end % transform(o, uparams, inverse)

    end % methods
    
end 

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

