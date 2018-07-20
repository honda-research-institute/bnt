clear all;
close all;
clc;

rng(5,'twister');

% read reference trajectories
ref_data_left = csvread('sumo/ref_trajectory_left_turn.csv', 1, 0);
ref_data_straight = csvread('sumo/ref_trajectory_straight.csv', 1, 0);
ref_data_right = csvread('sumo/ref_trajectory_right_turn.csv', 1, 0);

% Read and process the data from SUMO (training data and the corresponding labels)
input_data = csvread('sumo/30/data_truth_1000.csv');
input_label= csvread('sumo/30/label_1000.csv');

num_input_feature = 8;

num_features = 8;
num_intents = 3;
merge_labels = 0;
ss = 3 ;

[obs_data, hidden_data, hidden_data_overall] = process_data(input_data, input_label, num_input_feature, merge_labels);

% split the data to training and test set
train_size = 800;
test_size = 200;
obs_data_train = obs_data(1:train_size);
hidden_data_train = hidden_data(1:train_size);
hidden_data_train_overall = hidden_data_overall(1,1:train_size);
obs_data_test = obs_data(train_size+1:train_size + test_size);
hidden_data_test = hidden_data(train_size+1:train_size + test_size);
hidden_data_test_overall = hidden_data_overall(1,train_size+1:train_size+test_size);

% learn 3 different HMMs for each intent using manually labelled data 
% (for initialization)
[obs_data_l, hidden_data_l, obs_data_s, hidden_data_s, obs_data_r, hidden_data_r] =...
    cluster_data(obs_data_train, hidden_data_train, hidden_data_train_overall);

[initState_l, transmat_l, mu_l, Sigma_l] = gausshmm_train_observed(obs_data_l, hidden_data_l, 3);
[initState_s, transmat_s, mu_s, Sigma_s] = gausshmm_train_observed(obs_data_s, hidden_data_s, 3);
[initState_r, transmat_r, mu_r, Sigma_r] = gausshmm_train_observed(obs_data_r, hidden_data_r, 3);

% Number of mixtures
M = 5;
% Number of hidden states in each HMM
X = 3;

Sigma0_l = repmat(eye(num_features), [1 1 X M]);
Sigma0_s = repmat(eye(num_features), [1 1 X M]);
Sigma0_r = repmat(eye(num_features), [1 1 X M]);

mu0_l = rand(num_features, X, M);
mu0_s = rand(num_features, X, M);
mu0_r = rand(num_features, X, M);

mixmat0 = mk_stochastic(rand(X,M));

for i=1:num_intents
    mu0_l(:,:,i)= mu0_l(:,:,i) + mu_l; 
    mu0_s(:,:,i)= mu0_s(:,:,i) + mu_s; 
    mu0_r(:,:,i)= mu0_r(:,:,i) + mu_r; 
    Sigma0_l(:,:,:,i)= Sigma_l;
    Sigma0_s(:,:,:,i)= Sigma_s;
    Sigma0_r(:,:,:,i)= Sigma_r;
end

[LL2_l, prior2_l, transmat2_l, mu2_l, Sigma2_l, mixmat2_l] = mhmm_em(obs_data_l, initState_l, transmat_l, mu0_l, Sigma0_l, mixmat0,  'max_iter', 100);
[LL2_s, prior2_s, transmat2_s, mu2_s, Sigma2_s, mixmat2_s] = mhmm_em(obs_data_s, initState_s, transmat_s, mu0_s, Sigma0_s, mixmat0,  'max_iter', 100);
[LL2_r, prior2_r, transmat2_r, mu2_r, Sigma2_r, mixmat2_r] = mhmm_em(obs_data_r, initState_r, transmat_r, mu0_r, Sigma0_r, mixmat0,  'max_iter', 100);

onodes = 3; % observed node
bnet_l = build_GMM_HMM(num_features, X, M, ss, onodes, prior2_l, transmat2_l, mu2_l, Sigma2_l, mixmat2_l);
bnet_s = build_GMM_HMM(num_features, X, M, ss, onodes, prior2_s, transmat2_s, mu2_s, Sigma2_s, mixmat2_s);
bnet_r = build_GMM_HMM(num_features, X, M, ss, onodes, prior2_r, transmat2_r, mu2_r, Sigma2_r, mixmat2_r);

% perform inference
engine_l = filter_engine(hmm_2TBN_inf_engine(bnet_l));
engine_s = filter_engine(hmm_2TBN_inf_engine(bnet_s));
engine_r = filter_engine(hmm_2TBN_inf_engine(bnet_r));
% engine{end+1} = filter_engine(jtree_2TBN_inf_engine(bnet));

hnodes = [1]; %ysetdiff(1:ss, onodes);
[ref, predicted_intent, overall_intent, resolve_point, probs, probs_normal] = ...
    inference(engine_l, engine_s, engine_r, ss, onodes, hnodes, obs_data_test, hidden_data_test, merge_labels, X);
sub_int = X * 3;
cnf_sub_intents =  zeros(sub_int,sub_int);
for i=1:test_size
    i
    cnf_sub_intents = cnf_sub_intents + confusionmat(ref{i},predicted_intent{i}(2,:), 'Order', 1:9);
end

cnf_overall_intents =   confusionmat(hidden_data_test_overall, overall_intent, 'Order',[1, 2, 3]);
figure;
subplot(2,1,1);
truth = imagesc(hidden_data_test_overall);
title('Truth');
subplot(2,1,2);
prediction = imagesc(overall_intent);
title('Prediction');
plot_cnf(cnf_sub_intents, sub_int, 1)
plot_cnf(cnf_overall_intents, [0, 1, 2, 3], 1)
plot_resolve(resolve_point, overall_intent, ref_data_left, ref_data_straight, ref_data_right)


function [obs_data, hidden_data, hidden_data_overall] = process_data(input_data, input_label, num_features, merge_labels)
obs_data = cell(1,size(input_data,1));
hidden_data = cell(1,size(input_data,1));
hidden_data_overall = zeros(1,size(input_data,1));
for i=1:size(input_data,1)
    time_index = 1;
    for j=1:num_features:size(input_data,2)
        if ~all(input_data(i, j:j+num_features-1) == 0)
            obs_data{i}(1:num_features, time_index) = input_data(i, j:j+num_features-1);

            % feat_vector = input_data(i, j:j+num_features-1);
            % feat_vector = [feat_vector(1,1:4), feat_vector(1,7)];
            % obs_data{i}(1:length(feat_vector), time_index) = feat_vector;
            if merge_labels
                if input_label(i, time_index)==7marg_l
                    hidden_data{i}(1, time_index) = 1;
                elseif input_label(i, time_index)== 8 || input_label(i, time_index)== 9
                    hidden_data{i}(1, time_index) = input_label(i, time_index) - 1;
                else
                    hidden_data{i}(1, time_index) = input_label(i, time_index);
                end
            else
                hidden_data{i}(1, time_index) = input_label(i, time_index);
            end
            time_index = time_index + 1;
        else
            break
        end
    end
    if hidden_data{i}(1,end) == 3
        hidden_data_overall(1,i) = 1;
    elseif  hidden_data{i}(1,end) == 6
        hidden_data_overall(1,i) = 2;
    else
        hidden_data_overall(1,i) = 3;
    end
end
end


function [obs_data_l, hidden_data_l, obs_data_s, hidden_data_s, obs_data_r, hidden_data_r] = ...
    cluster_data(obs_data_train, hidden_data_train, hidden_data_train_overall)

num_left =  length(find(hidden_data_train_overall==1));
num_straight =  length(find(hidden_data_train_overall==2));
num_right = length(find(hidden_data_train_overall==3));

obs_data_l = cell(1,num_left);
obs_data_s = cell(1, num_straight);
obs_data_r = cell(1, num_right);

hidden_data_l = cell(1, num_left);
hidden_data_s = cell(1, num_straight);
hidden_data_r = cell(1, num_right);

l=1;
s=1;
r=1;
for i=1:length(obs_data_train)
    if hidden_data_train_overall(1,i) == 1
        obs_data_l{l} = obs_data_train{i};
        hidden_data_l{l}= hidden_data_train{i};
        l = l + 1;
    elseif hidden_data_train_overall(1,i) == 2
        obs_data_s{s} = obs_data_train{i};
        hidden_data_s{s}= hidden_data_train{i}-3;
        s = s + 1;
    elseif hidden_data_train_overall(1,i) == 3
        obs_data_r{r} = obs_data_train{i};
        hidden_data_r{r}= hidden_data_train{i}-6;
        r = r + 1;
    end
    
end
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

function [ref, predicted_intent, overall_intent, resolve_point, probs, probs_normal] = inference(engine_l, engine_s, engine_r, ss, onodes, hnodes, obs_data_test, hidden_data_test, merge_labels, X)
test_size = length(obs_data_test);
% define variables to store the results
predicted_intent = cell(1,test_size);
overall_intent = zeros(1,test_size);
resolve_point = -1 * ones(3,test_size);
probs = cell(1,test_size);
probs_normal = cell(1,test_size);
ref=cell(1,test_size);
threshold = 0.95;
wait_T = 4;
l_index = 1;
s_index = X+1;
r_index = X*2+1;
if merge_labels
    r_index = 7;
end
% do online ineference for each instance, at each given timestamp
for i=1:test_size
    i
    evidence = cell(ss,size(obs_data_test{i},2));
    ref{i} = hidden_data_test{i}(1,:);
    for t=1:size(obs_data_test{i},2)
        evidence{onodes,t} = obs_data_test{i}(:,t);
        [engine_l, ll_l] = enter_evidence(engine_l, evidence(:,t), t);
        marg_l = marginal_nodes(engine_l, hnodes, t);
        probs{i}(1:X,t) = marg_l.T;
        
        [engine_s, ll_s] = enter_evidence(engine_s, evidence(:,t), t);
        marg_s = marginal_nodes(engine_s, hnodes, t);
        probs{i}(X+1:X*2,t) = marg_s.T;
        
        [engine_r, ll_r] = enter_evidence(engine_r, evidence(:,t), t);
        marg_r = marginal_nodes(engine_r, hnodes, t);
        probs{i}(X*2+1:X*3,t) = marg_r.T;
        probs_normal{i}(:,t) = normalize(probs{i}(:,t));
        [ma,in]= max(probs_normal{i}(:,t));
        predicted_intent{i}(1:2,t) = [ma,in];
        if t >= wait_T+1 && resolve_point(1,i) < 0
            if all(probs_normal{i}(l_index,t-wait_T:t) >= threshold)  || all(probs_normal{i}(l_index+1,t-wait_T:t) >= threshold)|| all(probs_normal{i}(l_index+2,t-wait_T:t) >= threshold)
                overall_intent(1,i) = 1;
                resolve_point(1,i) = t;
                resolve_point(2,i) = obs_data_test{i}(1,t);
                resolve_point(3,i) = obs_data_test{i}(2,t);
            elseif all(probs_normal{i}(s_index,t-wait_T:t) >= threshold) || all(probs_normal{i}(s_index+1,t-wait_T:t) >= threshold) || all(probs_normal{i}(s_index+2,t-wait_T:t) >= threshold)
                overall_intent(1,i) = 2;
                resolve_point(1,i) = t;
                resolve_point(2,i) = obs_data_test{i}(1,t);
                resolve_point(3,i) = obs_data_test{i}(2,t);
            elseif all(probs_normal{i}(r_index,t-wait_T:t) >= threshold) || all(probs_normal{i}(r_index+1,t-wait_T:t) >= threshold) || all(probs_normal{i}(r_index+2,t-wait_T:t) >= threshold)
                overall_intent(1,i) = 3;
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

function plot_resolve(resolve_point, overall_intent, ref_data_left, ref_data_straight, ref_data_right)
figure;
hold on
plot(ref_data_left(:,1), ref_data_left(:,2),'-k')
plot(ref_data_straight(:,1), ref_data_straight(:,2),'-k')
plot(ref_data_right(:,1), ref_data_right(:,2),'-k')
for i=1:size(resolve_point,2)
    
    if overall_intent(1,i)==1
        plot(resolve_point(2, i)+200, resolve_point(3,i)+200, 'rx')
    elseif overall_intent(1,i)==2
        plot(resolve_point(2, i)+200, resolve_point(3,i)+200, 'bs')
    else
        plot(resolve_point(2, i)+200, resolve_point(3,i)+200, 'go')
    end
end
xlim([0, 400])
ylim([0, 400])
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
    set(gca, 'XTick', 1:X, ...
        'XTickLabel', {'L_{approach}', 'L_{pass}', 'L_{leave}','S_{approach}', 'S_{pass}', 'S_{leave}', 'R_{approach}', 'R_{pass}', 'R_{leave}'}, ...  %   and tick labels
        'YTick', 1:X, ...
        'YTickLabel',  {'L_{approach}', 'L_{pass}', 'L_{leave}', 'S_{approach}', 'S_{pass}', 'S_{leave}', 'R_{approach}', 'R_{pass}', 'R_{leave}'}, ...
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
