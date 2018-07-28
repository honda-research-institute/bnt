clear all;
% close all;
clc;

rng(50,'twister');

% read reference trajectories
ref_data_straight = readtable('~/repo/ngsim/csv/lankershim_207_mean.csv');
ref_data_right = readtable('~/repo/ngsim/csv/lankershim_208_mean.csv');

ref_data_straight = [ref_data_straight.Global_X ref_data_straight.Global_Y] .* 0.3048;
ref_data_right = [ref_data_right.Global_X ref_data_right.Global_Y] .* 0.3048;

% Read and process the data from NGSIM (training data and the corresponding labels)
num_features = 8;
num_input_feature = 8;
num_intents = 2;

input_data = readtable('~/repo/ngsim/csv/lankershim_filt.csv');
input_data = [input_data.Vehicle_ID input_data.D_Zone input_data.Global_X input_data.Global_Y input_data.v_Vel input_data.v_Acc];
input_data(:,3:end) = input_data(:,3:end) .* 0.3048;

merge_labels = 0;
X = num_intents * 2;

[obs_data, hidden_data, hidden_data_overall] = process_data(input_data, num_input_feature);

 
% split the data to training and test set
train_size = 107;
test_size = 27;
obs_data_train = obs_data(1:train_size);
hidden_data_train = hidden_data(1:train_size);
hidden_data_train_overall = hidden_data_overall(1,1:train_size);
obs_data_test = obs_data(train_size+1:train_size + test_size);
hidden_data_test = hidden_data(train_size+1:train_size + test_size);
hidden_data_test_overall = hidden_data_overall(1,train_size+1:train_size+test_size);

input_data = readtable('~/repo/ngsim/csv/selected_207.csv');
input_data1 = [input_data.Vehicle_ID input_data.D_Zone input_data.Global_X input_data.Global_Y input_data.v_Vel input_data.v_Acc];
input_data = readtable('~/repo/ngsim/csv/selected_208.csv');
input_data2 = [input_data.Vehicle_ID input_data.D_Zone input_data.Global_X input_data.Global_Y input_data.v_Vel input_data.v_Acc];
input_data_test = [input_data1; input_data2] ;
input_data_test(:,3:end) = input_data_test(:,3:end) .* 0.3048;
[obs_data_test, hidden_data_test, hidden_data_test_overall] = process_data(input_data_test, num_input_feature);

% learn the parameters of HMM using Maximum Likelihood
[initState, transmat, mu, Sigma] = gausshmm_train_observed(obs_data_train, hidden_data_train, X);

% Number of mixtures
M = 4;

Sigma0 = repmat(eye(num_features), [1 1 X M]);
mu0 = rand(num_features, X, M);
mixmat0 = mk_stochastic(rand(X,M));

for i=1:M
    mu0(:,:,i) = mu ;%+ mu0(:,:,i);
    Sigma0(:,:,:,i)= Sigma ;
  
end

[LL1, prior1, transmat1, mu1, Sigma1, mixmat1] = mhmm_em(obs_data_train, initState, transmat, mu0, Sigma0, mixmat0,  'max_iter', 100);
ss = 3 ;
onodes = 3; % observed node
bnet = build_GMM_HMM(num_features, X, M, ss, onodes, prior1, transmat1, mu1, Sigma1, mixmat1);

% perform inference
engine = {};
engine{end+1} = filter_engine(hmm_2TBN_inf_engine(bnet));


hnodes = [1]; %ysetdiff(1:ss, onodes);
[ref, predicted_intent, overall_intent, resolve_point, probs] = ...
    inference(engine, ss, onodes, hnodes, obs_data_test, hidden_data_test, merge_labels);

% cnf_sub_intents =  zeros(X,X);
% for i=1:test_size
%     cnf_sub_intents = cnf_sub_intents + confusionmat(ref{i},predicted_intent{i}(2,:), 'Order',1:X);
% end

cnf_overall_intents =   confusionmat(hidden_data_test_overall, overall_intent, 'Order',1:num_intents); 
figure;
subplot(2,1,1);
truth = imagesc(hidden_data_test_overall);
title('Truth');
subplot(2,1,2);
prediction = imagesc(overall_intent);
title('Prediction');
% plot_cnf(cnf_sub_intents, X, 1)
plot_cnf(cnf_overall_intents, num_intents, 1)
plot_resolve(resolve_point, overall_intent, ref_data_straight, ref_data_right)


function [obs_data, hidden_data, hidden_data_overall] = process_data(input_data, num_features)
center_x = 6452440.881 * 0.3048;
center_y = (1874188.179 +(1874259.679-1874188.179)/2) * 0.3048;
y_sublabel = 1874161.679 * 0.3048 - center_y ;

veh_IDs = unique(input_data(:,1));
num_cases = length(veh_IDs);
obs_data = cell(1,num_cases);
hidden_data = cell(1,num_cases);
hidden_data_overall = zeros(1,num_cases);
for i=1:num_cases
    obs = input_data(find(input_data(:,1) == veh_IDs(i,1)), :);
    hidden_data_overall(1,i) =  -(obs(1,2) - 209);

    obs = obs(:, 3:end)';
    obs = obs(:, find(obs(2,:) > (1874161.679*.3048-15)));
    obs = obs(:, find(obs(2,:) < (1874267.679*.3048)));
    obs = obs(:, find(obs(1,:) < (6452520.881*.3048)));
    obs(1,:) = obs(1,:) - center_x;
    obs(2,:) = obs(2,:) - center_y;
%     x = obs(1,:) - center_x;
%     y = obs(2,:) - center_y;
%     obs(1,:) = y;
%     obs(2,:) = -1 * x;
    dx = zeros(1, size(obs,2));
    dx(1,2:end) = obs(1, 2:end) - obs(1,1:end-1 );
    dy = zeros(1, size(obs,2));
    dy(1,2:end) = obs(2, 2:end) - obs(2,1:end-1 );
    angle = atan2(dy, dx);
    vx = obs(3,:) .* cos (angle);
    vy = obs(3,:) .* sin (angle);
    yaw = zeros(1, size(obs,2));
    yaw (1, 2:end) = (angle(1,2:end) - angle(1,1:end-1)) / 0.1;
    obs = [obs(1,:); obs(2,:); vx; vy; obs(3,:); obs(4,:); angle; yaw];
    obs_data{i} = obs;    
    if hidden_data_overall(1,i) == 1
        LS_ind = find(obs(2, :) >= y_sublabel);
        labels = ones(1,size(obs,2));
        labels(1, LS_ind) = 2;
        hidden_data{i} = labels;
    else
        LR_ind = find(obs(2, :) >= y_sublabel);
        labels = 3 .* ones(1, size(obs,2));
        labels(1, LR_ind) = 4;
        hidden_data{i} = labels;
    end
end
end

function bnet = build_HMM(num_features, X, ss, onodes)
% Make an HMM with cont observations
% X1 -> X2
% |     |
% v     v
% Y1    Y2
% discrete latent (intention) nodes and
% continuous observed ([x, y, v, psi]) nodes
intra = zeros(ss);
intra(1,2) = 1;
inter = zeros(ss);
inter(1,1) = 1;
Y = num_features; % num observable symbols
ns = [X Y];
dnodes = 1;
% onodes = 2;
eclass1 = [1 2];
eclass2 = [3 2];
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bnet.CPD{1} = tabular_CPD(bnet, 1);
bnet.CPD{3} = tabular_CPD(bnet, 3);
bnet.CPD{2} = gaussian_CPD(bnet, 2);
end

function bnet = build_GMM_HMM(O, Q, M, ss, onodes, prior, transmat, mu, Sigma, mixmat)
% build the HMM with
% discrete latent (intention) nodes and
% continuous observed ([x, y, v, psi]) nodes
% Make an HMM with mixture of Gaussian observations
%    Q1 ---> Q2
%  /  |   /  |
% M1  |  M2  |
%  \  v   \  v
%    Y1     Y2
% where Pr(m=j|q=i) is a multinomial and Pr(y|m,q) is a Gaussian

% Q: num hidden states
% O: size of observed vectorinference
% M: num mixture components per state
intra = zeros(ss);
intra(1,[2 3]) = 1;
intra(2,3) = 1;
inter = zeros(ss);
inter(1,1) = 1;

ns = [Q M O];
dnodes = [1 2];

eclass1 = [1 2 3];
eclass2 = [4 2 3];
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, ...
    'observed', onodes);

% prior0 = normalise(rand(Q,1));
% transmat0 = mk_stochastic(rand(Q,Q));
% mixmat0 = mk_stochastic(rand(Q,M));
% mu0 = rand(O,Q,M);
% Sigma0 = repmat(eye(O), [1 1 Q M]);
bnet.CPD{1} = tabular_CPD(bnet, 1, prior);
bnet.CPD{2} = tabular_CPD(bnet, 2, mixmat);
bnet.CPD{3} = gaussian_CPD(bnet, 3, 'mean', mu, 'cov', Sigma);
bnet.CPD{4} = tabular_CPD(bnet, 4, transmat);
end


function [ref, predicted_intent, overall_intent, resolve_point, probs] = inference(engine, ss, onodes, hnodes, obs_data_test, hidden_data_test, merge_labels)
test_size = length(obs_data_test);
% define variables to store the results
predicted_intent = cell(1,test_size);
overall_intent = zeros(1,test_size);
resolve_point = zeros(3,test_size);
probs = cell(1,test_size);
ref=cell(1,test_size);
threshold = 0.95;
wait_T = 5;

% l_index = 2;
s_index = 2;
r_index = 4;

% do online ineference for each instance, at each given timestamp
for i=1:test_size
    i
    evidence = cell(ss,size(obs_data_test{i},2));
    ref{i} = hidden_data_test{i}(1,:);
    for t=1:size(obs_data_test{i},2)
        evidence{onodes,t} = obs_data_test{i}(:,t);
        [engine{1}, ll(1)] = enter_evidence(engine{1}, evidence(:,t), t);
        % compute marginal likelihood of latent variable
        marg = marginal_nodes(engine{1}, hnodes, t);
        probs{i}(:,t) = marg.T;
        [ma,in]= max(probs{i}(:,t));
        predicted_intent{i}(1:2,t) = [ma,in];
        if t >= wait_T+1 && resolve_point(1,i) == 0
            if all(probs{i}(s_index,t-wait_T:t) >= threshold) || all(probs{i}(s_index-1,t-wait_T:t) >= threshold)
                overall_intent(1,i) = 1;
                resolve_point(1,i) = t;
                resolve_point(2,i) = obs_data_test{i}(1,t);
                resolve_point(3,i) = obs_data_test{i}(2,t);
            elseif all(probs{i}(r_index,t-wait_T:t) >= threshold) || all(probs{i}(r_index-1,t-wait_T:t) >= threshold)
                overall_intent(1,i) = 2;
                resolve_point(1,i) = t;
                resolve_point(2,i) = obs_data_test{i}(1,t);
                resolve_point(3,i) = obs_data_test{i}(2,t);
            end
            
        end
    end
    %     subplot(2,1,1)
    %     imagesc(ref{i})
    %     subplot(2,1,2)
    %     imagesc(predicted_intent{i}(2,:))
end
end

function plot_resolve(resolve_point, overall_intent, ref_data_straight, ref_data_right)
figure;
hold on
plot(ref_data_straight(:,1), ref_data_straight(:,2),'-k')
plot(ref_data_right(:,1), ref_data_right(:,2),'-k')
center_x = 6452440.881 * 0.3048;
center_y = (1874188.179 +(1874259.679-1874188.179)/2) * 0.3048;
for i=1:size(resolve_point,2)
    
    if overall_intent(1,i)==1
        plot(resolve_point(2, i)+center_x, resolve_point(3,i)+center_y, 'rx')
    elseif overall_intent(1,i)==2
        plot(resolve_point(2, i)+center_x, resolve_point(3,i)+center_y, 'bs')
    end
end
% xlim([0, 400])
% ylim([0, 400])
axis equal
legend
hold off
end
function plot_cnf(cnf_sub_intents, X, normalization)
figure;
if normalization
    cnf_sub_intents = normalize(cnf_sub_intents,2);
end

imagesc(cnf_sub_intents,'CDataMapping','scaled');
colorbar

textStrings = num2str(cnf_sub_intents(:), '%0.2f');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:X);
text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
if X > 3
%     set(gca, 'XTick', 1:X, ...
%         'XTickLabel', {'L_{approach}', 'L_{pass}', 'L_{leave}','S_{approach}', 'S_{pass}', 'S_{leave}', 'R_{approach}', 'R_{pass}', 'R_{leave}'}, ...  %   and tick labels
%         'YTick', 1:X, ...
%         'YTickLabel',  {'L_{approach}', 'L_{pass}', 'L_{leave}', 'S_{approach}', 'S_{pass}', 'S_{leave}', 'R_{approach}', 'R_{pass}', 'R_{leave}'}, ...
%         'TickLength', [0 0]);
set(gca, 'XTick', 1:X, ...
        'XTickLabel', {'LorR_{approach}', 'L_{pass}', 'L_{leave}','S', 'R_{pass}', 'R_{leave}'}, ...  %   and tick labels
        'YTick', 1:X, ...
        'YTickLabel',  {'LorR_{approach}', 'L_{pass}', 'L_{leave}', 'S', 'R_{pass}', 'R_{leave}'}, ...
        'TickLength', [0 0]);
else
    
    set(gca, 'XTick', 1:X, 'XTickLabel', {'L', 'S', 'R'}, ...  %   and tick labels
        'YTick', 1:X, 'YTickLabel',  {'L', 'S', 'R'}, ...
        'TickLength', [0 0]);
end

if normalization
    title('Normalized Confusion Matrix for Intents');
else
    title('Non-Normalized Confusion Matrix for Intents');
end
end
