function gen_figures(m, results_smc, results_cwieki, results_ieki, modelname)
%% display table with the number of likelihood mean evaluations (G)

n = length(results_ieki);
IEKI_labels = strings(1, n);
penalties = zeros(1, n + 2);
penalties(1) = results_smc.penalty; 
penalties(2) = results_cwieki.penalty;
for i = 1:n
    penalties(2 + i) = results_ieki{i}.penalty;
    IEKI_labels(i) = "IEKI ($\sigma = " + sprintf("%.2f", results_ieki{i}.phi) + "$)";
end
relpen = round(penalties(1)./penalties, 1);

% print results
format_str = ['%.1f' repmat(' & %.1f', 1, n+1) ' \\\\ \n'];
fprintf('\n');
fprintf("$G(\\cdot)$ evaluations & " + format_str, penalties); 
fprintf("Approximate speed-up & " + format_str, relpen); 
fprintf('\n');

%% plot densities

nrows = ceil(m.np/4);
ncols = min(m.np, 4);

figure('Position', [430,400,1000,250*nrows]);
tiledlayout(nrows, ncols);

for i = 1:m.np 
    nexttile
    hold on

    % SMC results
    [f,xi] = ksdensity(results_smc.samples(:, i), 'support', m.param_support{i}, 'BoundaryCorrection', 'Reflection');
    plot(xi,f,'-','LineWidth',2);
    xlims = xlim;

    % CW-IEKI results
    [f,xi] = ksdensity(results_cwieki.samples(:, i), 'support', m.param_support{i}, 'BoundaryCorrection', 'Reflection');
    plot(xi,f,'--','LineWidth',2);
    
    
    % IEKI results
    if i <= m.theta_block(end)
        for j = 1:n
            [f,xi] = ksdensity(results_ieki{j}.samples(:, i), 'support', m.param_support{i}, 'BoundaryCorrection', 'Reflection');
            plot(xi,f,'-','LineWidth',1);
        end
    end

    % prior density
    x_vals = linspace(xlims(1), xlims(2), 1000);
    y_vals = exp(m.priori_lpdf(x_vals, i));
    plot(x_vals, y_vals', 'Color', 'k', 'LineWidth', 2)

    % labels
    xlim(xlims); 
    title(m.names{i}, 'FontSize', 12, 'interpreter','latex');
    xline(m.theta(i));
    
    if i == 1
        lg = legend(["SMC", "CW-IEKI", IEKI_labels, "Prior", "True"], 'Box', 'off', 'interpreter','latex'); 
        lg.Layout.Tile = 'East';
        lg.FontSize = 12;
    end
end

print(gcf,"figures/" + modelname + "_densities.eps",'-depsc2','-r300');

end