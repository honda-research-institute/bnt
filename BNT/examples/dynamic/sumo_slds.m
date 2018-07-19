close all;
clear all;
clc;


rng(50,'twister');

% Read and process the data from SUMO (training data and the corresponding labels)
ref_data_straight = csvread('sumo/ref_trajectory_straight.csv', 1, 0);
ref_data_right = csvread('sumo/ref_trajectory_right_turn.csv', 1, 0);

input_data = csvread('sumo/30/data_truth_10.csv');
input_label= csvread('sumo/30/label_10.csv');
truth_data = csvread('sumo/30/data_truth_10.csv');

train_size = 0;
test_size = 10;

obs_data = cell(1,size(input_data,1));
hidden_data = cell(1,size(input_data,1));
hidden_data_overall = zeros(1,size(input_data,1));
truth = cell(1,size(truth_data,1));

Z = 2; % num hidden states
X = 4; 
Y = 4; % num observable symbols

for i=1:size(input_data,1)
    time_index = 1;
    for j=1:4:size(input_data,2)
        if ~all(input_data(i, j:j+3) == 0)
            obs_data{i}(1:Y, time_index) = input_data(i, j:j+Y-1);
            truth{i}(1:X, time_index) = truth_data(i, j:j+X-1);
            if input_label(i, time_index) < 4
                hidden_data{1,i}{1,time_index}= 1; %input_label(i, time_index);
            else 
                hidden_data{1,i}{1,time_index}= 2;
            end
            hidden_data{1,i}{2,time_index} = truth{i}(:,time_index);
            time_index = time_index + 1;
        else
            break
        end
    end
    if hidden_data{1,i}{1,1} == 1
        hidden_data_overall(1,i) = 1;
    else
        hidden_data_overall(1,i) = 2;
    end
end

% split the data to training and test set
obs_data_train = obs_data(1:train_size);
hidden_data_train = hidden_data(1:train_size);
hidden_data_train_overall = hidden_data_overall(1,1:train_size);

num_straight = length(find(hidden_data_train_overall(:)==1));
num_right = length(find(hidden_data_train_overall(:) == 2));
obs_data_s = cell(1,size(num_straight,1));
obs_data_r = cell(1,size(num_right,1));
s_i = 1;
r_i = 1;
for i=1:length(hidden_data_train_overall)
   if  hidden_data_train_overall(1,i) == 1
       obs_data_s{s_i} = obs_data_train{i};
       s_i = s_i + 1;
   else
       obs_data_r{r_i} = obs_data_train{i};
       r_i = r_i + 1;
   end
end

% build the SLDS with
% discrete latent (intention) nodes and
% continuous latent ([x, y, v, psi, a, delta])
% continuous observed ([x, y, v, psi]) nodes
n = 3;
intra = zeros(n);
intra(1,[2, 3]) = 1;
intra(2, 3) = 1;
inter = zeros(n);
inter(1,1) = 1;
inter(2,2) = 1;

ns = [Z X Y];
dnodes = 1;
onodes = 3;
eclass1 = [1 2 3];
eclass2 = [4 5 3];
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2, 'observed', onodes);
bnet.CPD{1} = tabular_CPD(bnet, 1);
bnet.CPD{2} = gaussian_CPD(bnet, 2);
bnet.CPD{3} = gaussian_CPD(bnet, 3);
bnet.CPD{4} = tabular_CPD(bnet, 4);
bnet.CPD{5} = gaussian_CPD(bnet, 5);

% % this section is for EM
% obs_data_train_cell = cell(1,train_size);
% for i=1:length(obs_data_train)
%     obs_data_train_cell{1,i} = cell(3,length(obs_data_train{i}));
%     for j =1:length(obs_data_train{i})
%         obs_data_train_cell{1,i}{1,j} = hidden_data_train_overall(1,i);
%         obs_data_train_cell{1,i}{2,j} = truth{i}(1:X,j);
%         obs_data_train_cell{1,i}{3,j} = obs_data_train{i}(:,j);
% 
%     end
% end
% 
% ss = length(bnet.intra);
% par.A = zeros(X,X,Z);
% par.B = zeros(X,X,Z);
% par.C = zeros(Y,X,Z);
% par.D = zeros(Y,Y,Z);
% par.Q = zeros(X,X,Z);
% par.R = zeros(Y,X,Z);
% par.mu0 = zeros(X,1,Z);
% par.S0 = zeros(X,X,Z);
% max_iter = 20;
% 
% for i=1:Z
%    par.A(:,:,i) = 1.0 * eye(X, X);
%    par.C(:,:,i) = 1.0 * eye(Y, X);
%    par.Q(:,:,i) = 1.0 * eye(X, X);
%    par.R(:,:,i) = 0.5 * eye(Y, X);
%    par.mu0(:,:,i) = [-25, -1, 20, 0]';
%    par.S0(:,:,i) = eye(X, X);
%    if i == 1
%        data = obs_data_s;
%    else
%        data = obs_data_r;
%    end
%    
%    [par.A(:,:,i), par.C(:,:,i), par.Q(:,:,i), par.R(:,:,i), par.mu0(:,:,i), par.S0(:,:,i), LL(:,:,i)] = ...
%     learn_kalman(data,par.A(:,:,i), par.C(:,:,i), par.Q(:,:,i), par.R(:,:,i), par.mu0(:,:,i), par.S0(:,:,i), max_iter);
% 
% end
% 
% save('results')
N = 500;
n_x = X;
n_y = Y;
n_z = Z;
par.A = zeros(n_x,n_x,n_z);
par.B = zeros(n_x,n_x,n_z);
par.C = zeros(n_y,n_x,n_z);
par.D = zeros(n_y,n_y,n_z);
par.E = zeros(n_x,n_x,n_z);
par.F = zeros(n_x,1,n_z);
par.G = zeros(n_y,1,n_z);
TT = 0.1; % sampling period
for i=1:n_z,
  par.A(:,:,i) = [1 TT 0 0; 0 1 0 0; 0 0 1 TT; 0 0 0 1];
  par.C(:,:,i) = eye(n_x);
  par.B(:,:,i) = 0.2*eye(n_x,n_x);    
  par.D(:,:,i) = 3*diag([2,1,2,1]); % sqrt(3)*diag([20,1,20,1]);    
  %par.F(:,:,i) = 0;
  par.G(:,:,i) = zeros(n_y,1);   
end;
% input / control vectors
par.F(:,1,1)  = [0 0 0 0]';
par.F(:,1,2) = [1.225, 0.35, -1.225, -0.35]';

% Markov chain
par.mu0 = zeros(n_x,1);                 % Initial Gaussian mean.
par.S0  = 1*eye(n_x,n_x);             % Initial Gaussian covariance.  

par.pz0 = [0 1];
par.T = [1.0 0.0; 0.0, 1.0];
%%                          GENERATE THE DATA
obs_data_test = obs_data(train_size+1:train_size + test_size);
hidden_data_test = hidden_data(train_size+1:train_size + test_size);
hidden_data_test_overall = hidden_data_overall(1,train_size+1:train_size+test_size);

%% Estimation
% zest_pf = cell(1, test_size);
% xest_pf = cell(1, test_size);
% zamples_pf = cell(1, test_size);
% xsamples_pf = cell(1, test_size);
% tic;
% for i=1:10
%     u = ones(1,length(obs_data_test{i}));
%     [zest_pf{i}, xest_pf{i}, zamples_pf{i}, xsamples_pf{i}] = pfSlds(N, par, obs_data_test{i}, u);
% end
% time_pf = toc   

zest_rbpf = cell(1, test_size);
xest_rbpf = cell(1, test_size);
zsamples_rbpf = cell(1, test_size);
tic;
for i=1:10
    u = ones(1,length(obs_data_test{i}));
    [zest_rbpf{i}, xest_rbpf{i}, zsamples_rbpf{i}] = rbpfSlds(N, par, obs_data_test{i}, u);
end
time_rbpf = toc  
% engine_learn = jtree_dbn_inf_engine(bnet);
% [bnet, LL, engine_learn] = learn_params_dbn_em(engine_learn, obs_data_train_cell, 'max_iter', 7);
% 
% % build the inference engines
% engine = {};
% engine{end+1} = filter_engine(jtree_dbn_inf_engine(bnet));
% ss = length(bnet.intra);
% E = length(engine);
% hnodes = mysetdiff(1:ss, onodes);
% 
% obs_data_test = obs_data(train_size+1:train_size + test_size);
% hidden_data_test = hidden_data(train_size+1:train_size + test_size);
% hidden_data_test_overall = hidden_data_overall(1,train_size+1:train_size+test_size);
% 
% % define variables to store the results
% predicted_intent = cell(1,test_size);
% overall_intent = zeros(1,test_size);
% resolve_point = zeros(3,test_size);
% probs = cell(1,test_size);
% ref=cell(1,test_size);
% threshold = 0.95;
% wait_T = 4;
% % do online ineference for each instance, at each given timestamp
% for i=1:test_size
%     i
%     evidence = cell(ss,size(obs_data_test{i},2));
%     ref{i} = hidden_data_test{i}(1,:);
%     for t=1:size(obs_data_test{i},2)
%         evidence{onodes,t} = obs_data_test{i}(:,t);
%         [engine{1}, ll(1)] = enter_evidence(engine{1}, evidence(:,t), t);
%         % compute marginal likelihood of latent variable
%         marg = marginal_nodes(engine{1}, 2, t);
%         probs{i}(:,t) = marg.T;
%         [ma,in]= max(probs{i}(:,t));
%         predicted_intent{i}(1:2,t) = [ma,in];
%         if t >= wait_T+1 && resolve_point(1,i) == 0
% %             if all(probs{i}(1,t-wait_T:t) >= threshold) || all(probs{i}(2,t-wait_T:t) >= threshold)
% %                 overall_intent(1,i) = 1;
% %                 resolve_point(1,i) = t;
% %                 resolve_point(2,i) = obs_data_test{i}(1,t);
% %                 resolve_point(3,i) = obs_data_test{i}(2,t);
% %             elseif all(probs{i}(4,t-wait_T:t) >= threshold) || all(probs{i}(4,t-wait_T:t) >= threshold)
% %                 overall_intent(1,i) = 2;
% %                 resolve_point(1,i) = t;
% %                 resolve_point(2,i) = obs_data_test{i}(1,t);
% %                 resolve_point(3,i) = obs_data_test{i}(2,t);
% %             end
%             
%         end
%     end
%     subplot(2,1,1)
%     imagesc(ref{i})
%     subplot(2,1,2)
%     imagesc(predicted_intent{i}(2,:))
% end
% % % hold on;
% % % hold off;
% % 
% % cnf_sub_intents =  zeros(X,X);
% % for i=1:test_size
% %     cnf_sub_intents = cnf_sub_intents + confusionmat(ref{i},predicted_intent{i}(2,:), 'Order',[1 2 3 4 5 6]);
% % end
% % 
% % cnf_overall_intents =   confusionmat(hidden_data_test_overall, overall_intent, 'Order', [1 2]);
% % 
% % figure;
% % subplot(2,1,1);
% % truth = imagesc(hidden_data_test_overall);
% % title('Truth');
% % subplot(2,1,2);
% % prediction = imagesc(overall_intent);
% % title('Prediction');
% % plot_cnf(cnf_sub_intents, X, 1)
% % plot_cnf(cnf_overall_intents, 2, 1)
% % plot_resolve(resolve_point, overall_intent, ref_data_straight, ref_data_right)
% % 
% % function plot_resolve(resolve_point, overall_intent, ref_data_straight, ref_data_right)
% %     figure;
% %     hold on
% %     plot(ref_data_straight(:,1), ref_data_straight(:,2),'-k')
% %     plot(ref_data_right(:,1), ref_data_right(:,2),'-k')
% %     for i=1:size(resolve_point,2)
% % 
% %         if overall_intent(1,i)==1
% %             plot(resolve_point(2, i)+200, resolve_point(3,i)+200, 'rx')
% %         else
% %             plot(resolve_point(2, i)+200, resolve_point(3,i)+200, 'go')
% %         end
% %     end
% %     xlim([0, 400])
% %     ylim([0, 400])
% %     legend
% %     hold off
% % end
% % function plot_cnf(cnf_sub_intents, X, normalization)
% %     figure;
% %     if normalization
% %         cnf_sub_intents = normalize(cnf_sub_intents,2);
% %     end
% %         
% %     imagesc(cnf_sub_intents,'CDataMapping','scaled');
% %     colorbar
% % 
% %     textStrings = num2str(cnf_sub_intents(:), '%0.2f');      
% %     textStrings = strtrim(cellstr(textStrings));
% %     [x, y] = meshgrid(1:X); 
% %     text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
% %     if X > 2
% %         set(gca, 'XTick', 1:X, ...              
% %         'XTickLabel', {'S_{approach}', 'S_{pass}', 'S_{leave}', 'R_{approach}', 'R_{pass}', 'R_{leave}'}, ...  %   and tick labels
% %         'YTick', 1:X, ...
% %         'YTickLabel',  {'S_{approach}', 'S_{pass}', 'S_{leave}', 'R_{approach}', 'R_{pass}', 'R_{leave}'}, ...
% %         'TickLength', [0 0]);
% %     else
% %         set(gca, 'XTick', 1:X, 'XTickLabel', {'S', 'R'}, ...  %   and tick labels
% %         'YTick', 1:X, 'YTickLabel',  {'S','R'}, ...
% %         'TickLength', [0 0]);
% %     end
% %     
% %     if normalization
% %         title('Normalized Confusion Matrix for Intents');
% %     else
% %         title('Non-Normalized Confusion Matrix for Intents');
% %     end
% % end
