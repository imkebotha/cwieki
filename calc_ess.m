function ess = calc_ess(newgamma, oldgamma, part_w, part_loglike)

log_part_w = log(part_w) + (newgamma - oldgamma)*part_loglike;

%numerically stabilize before exponentiating
log_part_w = log_part_w - max(log_part_w);
part_w = exp(log_part_w);

%normalize weights
part_w = part_w/sum(part_w);

% compute effective sample size
ess = 1/sum(part_w.^2);

end

