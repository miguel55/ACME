clear, close all;
config;
% Dir instantaneous and dynamic features, labels and tracking results
dF=dir(['results/*/*/' id_features '.mat']);
dL=dir(['results/*/*/' id_label '.mat']);
dT=dir(['results/*/*/' id_features '_track.mat']);
dH=dir('results/*/*/handles.mat');
% Cell feature, labels and identifiers
cell_features=[];
cell_labels=[];
cell_tracks=[];
cells_by_capture=[];
cells_by_volume=[];
cells_by_timestamp=[];
cells_by_trajectory=[];
cells_by_id=[];
% Load the information
id_trajs=0;
for i=1:length(dF)
    aux=strsplit(dF(i).folder,'/'); 
    aux2=strsplit(aux{end},'_');
    load([dF(i).folder '/' dF(i).name]);
    load([dL(i).folder '/' dL(i).name]);
    load([dT(i).folder '/' dT(i).name]);
    load([dH(i).folder '/' dH(i).name]);
    cell_features=[cell_features; features_simple(labels==1,:)];
    cell_tracks=[cell_tracks; features_track(labels==1,:)];
    cell_labels=[cell_labels; str2double(aux{end-1}(end))*ones(sum(labels==1),1)];
    cells_by_capture=[cells_by_capture; i*ones(size(features_simple(labels==1,:),1),1)];
    cells_by_volume=[cells_by_volume; str2double(aux2{1})*ones(size(features_simple(labels==1,:),1),1)];
    cells_by_timestamp=[cells_by_timestamp; handles.nodeNetwork(labels==1,5)];
    cells_by_id=[cells_by_id; handles.nodeNetwork(labels==1,6)];
    ids=handles.nodeNetwork(labels==1,6);
    trajs=[];
    for j=1:length(ids)
        [x,y]=find(handles.finalNetwork==ids(j));
        trajs=[trajs; y];
    end
    trajs_unique=unique(trajs);
    trajs_new=trajs;
    for j=1:length(trajs_unique)
        trajs_new(trajs==trajs_unique(j))=id_trajs+j;
    end
    cells_by_trajectory=[cells_by_trajectory; trajs_new];
    id_trajs=id_trajs+length(trajs_unique);
end

% Extract the useful features and generate train matrices
XSIMPLE = cell_features(:,useful_features_inst(1:length(useful_features_inst)-1));
XTRACK = cell_tracks(:,useful_features_dyn);
y = cell_labels;
X_train=[XTRACK XSIMPLE];
feat_unnorm=[XTRACK XSIMPLE]; % Features without normalization
y_train=y;

% Data normalization
mus=mean(X_train(ismember(y_train,groups_beh_discovery),:),1);
stds=std(X_train(ismember(y_train,groups_beh_discovery),:),1);
X_train=(X_train-repmat(mus,[size(X_train,1) 1]))./repmat(stds,[size(X_train,1) 1]);

%% Non-supervised cell behavior discovery
criterionBD=zeros(nK,1);
criterionBD(1)=1;
X_train12=X_train(ismember(y_train,groups_beh_discovery),:);
y_train12=y_train(ismember(y_train,groups_beh_discovery));
for K=2:nK
    criterion_aux=zeros(nK-1,size(X_train12,2));
    rng(seed);
    [cidx2,cmeans2,sumD,d] = kmeans(X_train12,K,'dist','sqeuclidean'); %,'Replicates',10
    % For each group, build the behavior histogram that indicates how many 
    % cells present each behavior)
    histograms=zeros(length(groups_beh_discovery),K);
    for i=1:length(groups_beh_discovery)
        cluster=cidx2(y_train12==groups_beh_discovery(i));
        histograms(i,:)=hist(cluster,1:K)/length(cluster); 
    end
    % K_opt selection criterion: group histogram intersection. If it reduces
    % too much if a new behavior is included, is relevant
    hist_intersection=Inf*ones(length(groups_beh_discovery));
    for i=1:length(groups_beh_discovery)
        for j=1:length(groups_beh_discovery)
            % Histogram intersection
            hist_intersection(i,j)=sum(min([histograms(i,:);histograms(j,:)],[],1));
        end
    end
    hist_intersection=triu(hist_intersection)-2*tril(ones(length(groups_beh_discovery)));
    % Kopt: maximum of histogram intersection (worst case)
    criterionBD(K)=max(hist_intersection(hist_intersection(:)>0));
end
% Graphical representation
figure; plot(1:nK,criterionBD), xlabel('K (number of behaviors)');
ylabel('Maximum IH');
title('Criterion. Histogram intersection maximum for each group (worst case).'); grid on;
criterionBD(~islocalmin(criterionBD))=1;
[~,K_selected]=min(criterionBD);
% K-Means (behavior discovery)
cidx=zeros(size(y_train));
rng(seed);
[cidx12,cmeans] = kmeans(X_train12,K_selected,'dist','sqeuclidean','Replicates',100);
cidx(y_train<(nG_BD+1))=cidx12;
X_trainR=X_train(y_train>=(nG_BD+1),:);
ds=pdist2(X_trainR,cmeans);
[ds,p]=min(ds,[],2);
cidx(y_train>=(nG_BD+1))=p;
%% Hierarchical explainability
K_exp=K_selected;
X_trainH=[XTRACK XSIMPLE];
% First: hierarchy generation
valid=true;
hierarchy=cell(0);
todolist=cell(1);
todolist{1}=ones(1,K_exp);
while (~isempty(todolist))
    if (sum(todolist{1}(todolist{1}>0))>1)
        % Analyze the clusters
        subset=ismember(cidx,find(todolist{1}));
        clust=cidx(subset);
        clust_remap=zeros(size(clust));
        idU=unique(clust);
        for i=length(idU):-1:1
            clust_remap(clust==idU(i))=i;
        end
        % Obtain the best division for this level of the hierarchy
        [valid,C]=get_best_division(X_trainH(subset,:),clust_remap,sum(todolist{1}(todolist{1}>0)));
        % Store the current division and update the to-do-list
        if (valid)
            todolist{end+1}=-1*ones(size(todolist{1}));
            todolist{end}(todolist{1}==1)=double(C==1);
            todolist{end+1}=-1*ones(size(todolist{1}));
            todolist{end}(todolist{1}==1)=double(C==0);
            hierarchy{end+1}=-1*ones(size(todolist{1}));
            if (sum(double(C==0))<sum(double(C==1)))
                hierarchy{end}(todolist{1}==1)=C==0;
            else
                hierarchy{end}(todolist{1}==1)=C==1;
            end
        else
            if (sum(todolist{1}(todolist{1}>0))>2)
                offset=0;
            else
                offset=1;
            end
            for j=1:sum(todolist{1}(todolist{1}>0))-offset
                hierarchy{end+1}=-1*ones(size(todolist{1}));
                aux=zeros(size(C));
                aux(j)=1;
                hierarchy{end}(todolist{1}==1)=aux;
            end
        end
    end
    todolist(1)=[];
end

% Second: L1 classifier
lambda = logspace(-3,0,101);
accs = zeros(length(lambda),length(hierarchy));
numNZCoeff = zeros(length(lambda),length(hierarchy));
lims = zeros(1,length(hierarchy));
lambdas_selected=zeros(1,length(hierarchy));
important_features=cell(length(hierarchy),1);
% Code the hierarchy
hierarchy_mat=cell2mat(hierarchy');
coding=zeros(size(hierarchy_mat));
coding(hierarchy_mat==1)=1;
coding(hierarchy_mat==0)=-1;
coding(hierarchy_mat==-1)=0;
rng(seed);
% Fit the ECOC classifier with Lasso regularization
MdlFULL = fitcecoc(X_train',cidx','ObservationsIn','columns',...
    'Learner',templateLinear('Lambda',lambda,'learner','logistic','Regularization','lasso',...
    'Solver','sparsa'),'Coding',coding','PredictorNames',{feature_names{:}});%'FitBias',false
[y_pred,score_pred]=predict(MdlFULL,X_train);
for k=1:length(hierarchy)
    % Analyze for each lambda
    for i=1:length(lambda)
        [~,~,~,accs(i,k)] = perfcurve(cidx,score_pred(:,k,i),k);
        numNZCoeff(i,k) = sum(MdlFULL.BinaryLearners{k}.Beta(:,i)~=0);
    end
    lims(k)=find(accs(:,k)<FAITHFULNESS_LEVEL*accs(1,k),1,'first');
    lambdas_selected(k)=lambda(lims(k));
    features_selected=find(MdlFULL.BinaryLearners{k}.Beta(:,lims(k))~=0);
    % Get the most important features by applying the weights to them
    weights=MdlFULL.BinaryLearners{k}.Beta(features_selected,lims(k));
    [~,ids]=sort(abs(sum(repmat(weights',[size(X_train,1),1]).*X_train(:,features_selected),1)),'descend');
    weights=weights(ids);
    features_selected=features_selected(ids);
%     % Apply Feed-Forward Feature Selection if the number of features is
%     % high
    if (FFFS)
        if (length(features_selected)>3)
            criterion='diaglinear';
            fun = @(XT,yT,Xt,yt)classifySamples(XT,yT,Xt,yt,criterion);
            % 2-Fold Cross-Validation
            rng(seed);
            M=2;
            [train_ids,val_ids]=getCrossVal(size(X_train,1),M);
            error=0;
            X_train_aux=X_train(:,features_selected);
            subset1=find(hierarchy{k}==1);
            improvement=Inf;
            % Select the features that provide more than 3% of improvement
            tolFun=0.03;
            [~,ids_feats]=sort(sum(abs(repmat(weights',[size(X_train,1),1]).*X_train(:,features_selected))),'descend');
            X_already=X_train(:,ids_feats(1));
            features=ids_feats(1)';
            for j=1:length(train_ids)
                error=error+fun(X_already(train_ids{j},:),ismember(cidx(train_ids{j}),subset1),X_already(val_ids{j},:),ismember(cidx(val_ids{j}),subset1));
            end
            error=error/M;
            prev_error=error;
            while (improvement>tolFun)
                partial_error=zeros(size(X_train_aux,2),1);
                for m=1:length(features_selected)
                    for j=1:length(train_ids)
                        partial_error(m)=partial_error(m)+fun([X_already(train_ids{j},:) X_train_aux(train_ids{j},m)],ismember(cidx(train_ids{j}),subset1),[X_already(val_ids{j},:) X_train_aux(val_ids{j},m)],ismember(cidx(val_ids{j}),subset1));
                    end
                end
                partial_error=partial_error/M;
                partial_error(features)=Inf;
                [error_it,sel]=min(partial_error);
                improvement=prev_error-error_it;
                if (improvement>tolFun)
                    prev_error=error_it;
                    error=[error; error_it];
                    features=[features; sel];
                    X_already=[X_already X_train_aux(:,sel)];
                end
            end
            weights=weights(features);
            features_selected=features_selected(features);
        end
    end
    % Generate the textual description of the hierarchy
    important_features{k}={feature_names{features_selected}};
    for j=1:length(features_selected)
        if (weights(j)<0)
            important_features{k}{j}={['- ' important_features{k}{j} ' ' num2str(abs(weights(j))/sum(abs(weights)))]};
        else
            important_features{k}{j}={['+ ' important_features{k}{j} ' ' num2str(abs(weights(j))/sum(abs(weights)))]};
        end
    end
end



%% Gaussian explicability: for feature importance. We assume the features
%% have a Gaussian distribution and we compute the overlaps between each
%% pair of behaviors
ovl=zeros(size(X_train,2),K_selected,K_selected);
for i=1:K_selected
    feat_intra_mus=mean(X_train(cidx==i,:),1);
    feat_intra_stds=std(X_train(cidx==i,:),1);
    for j=1:K_selected
        if (i~=j)
            feat_inter_mus=mean(X_train(cidx==j,:),1);
            feat_inter_stds=std(X_train(cidx==j,:),1);
            [mu1,p1]=min([feat_intra_mus; feat_inter_mus],[],1);
            [mu2,p2]=max([feat_intra_mus; feat_inter_mus],[],1);
            aux1=[feat_intra_stds; feat_inter_stds];
            aux2=[feat_intra_stds; feat_inter_stds];
            sigma1=zeros(size(mu1));
            sigma2=zeros(size(mu2));
            for m=1:size(sigma1,2)
                sigma1(m)=aux1(p1(m),m);
                sigma2(m)=aux2(p2(m),m);
            end
            % Numerical OVL
            Npoints=10000;
            for k=1:size(ovl,1)
                xmin=min([mu1(k)-6*sigma1(k),mu2(k)-6*sigma2(k)]);
                xmax=max([mu1(k)+6*sigma1(k),mu2(k)+6*sigma2(k)]);
                ovl(k,i,j)= calc_overlap_twonormal(sigma1(k),sigma2(k),mu1(k),mu2(k),xmin,xmax,(xmax-xmin)/(Npoints+1));
                if (isnan(ovl(k,i,j)) )
                    % If stds are 0
                    if (mu1(k)==mu2(k))
                        ovl(k,i,j)=1;
                    else
                        ovl(k,i,j)=0;
                    end
                end
            end
        else
            ovl(:,i,j)=-Inf;
        end
    end
end 

% Encontrar para cada grupo las características que mejor separan los
% comportamientos
importance_complem=zeros(size(X_train,2),length(hierarchy));
for j=1:size(X_train,2)
    ovl_feat=squeeze(ovl(j,:,:));
    for i=1:length(hierarchy)
        subset1=find(hierarchy{i}==1);
        ovl_feat_group=ovl_feat(subset1,:);
        subset2=find(hierarchy{i}==0);
        ovl_feat_group=ovl_feat_group(subset2);
        % Worst case
        [importance_complem(j,i),worst_case_p]=max(ovl_feat_group);
    end
end

%% Visualization: behavior proportion in groups
id_group=categorical(y,unique(y),group_def);
behavior_aux=cell(1,K_selected);
for i=1:K_selected
    behavior_aux{1,i}=['Behaviour ',num2str(i)];
end
id_behavior=categorical(cidx,unique(cidx),behavior_aux);
sym_group_def='xos+*d^';
color_behavior_aux='rgbkcym';
sym_group='';
color_behavior='';
for i=1:nG
    for j=1:K_selected
        sym_group=[sym_group,sym_group_def(i)];
        color_behavior=[color_behavior,color_behavior_aux(j)];
    end
end 
% Distribuciones de cada comportamiento por grupo
histograms=zeros(nG,K_selected);
for i=1:nG
    cluster=cidx(y==i);
    histograms(i,:)=hist(cluster,1:K_selected)/length(cluster); 
end
figure;
bar(histograms,'stacked'); grid on;
title('Behavior distribution per groups');
xticks(1:nG); xticklabels(group_def);
ylabel('Behavior proportion');
leg_K=cell(1,K_selected);
for i=1:K_selected
    leg_K{1,i}=['Behavior ',num2str(i)];
end
legend(leg_K);

%% Print data
for i=1:length(hierarchy)
    ids1=find(hierarchy{i}==1);
    ids2=find(hierarchy{i}==0);
    disp(['Behaviors ',num2str(ids1),' respect to behaviors ',num2str(ids2),'. Description:']);
    for j=1:length(important_features{i})
        disp(important_features{i}{j});
    end
end
%% Save data
% Features, behaviors and groups
X=[feat_unnorm y cidx cells_by_volume cells_by_capture cells_by_trajectory cells_by_timestamp];
% CSV: for Python visualization
csvwrite('../data/extracted_cell_data.csv',X);
% Excel (human-readable)
filename = '../data/extracted_cell_data.xlsx';
feature_description=feature_names;
feature_description{end+1}='Group id';
feature_description{end+1}='Behavior id';
feature_description{end+1}='Volume id';
feature_description{end+1}='Capture id';
feature_description{end+1}='Trajectory id';
feature_description{end+1}='Timestamp';

for i=1:length(feature_description)
    feature_description{i}=strrep(feature_description{i},' ','_');
    feature_description{i}=strrep(feature_description{i},'(','_');
    feature_description{i}=strrep(feature_description{i},')','_');
    feature_description{i}=strrep(feature_description{i},'/','_');
end
writetable(array2table(X,'VariableNames',feature_description),filename,'Sheet',1);
% Feature importance
importance=1-importance_complem;
% CSV: for Python visualization
csvwrite('../data/feature_importance_hierarchical.csv',importance);
% Excel (human-readable)
filename = '../data/feature_importance_hierarchical.xlsx';
feature_description=cell(length(hierarchy),1);
for i=1:length(feature_description)
    feature_description{i}="Hierarchy_level_"+num2str(i);
end
writetable(array2table(importance,'VariableNames',cellstr(feature_description)),filename,'Sheet',1);

% Function that calculates the OVL between 2 normal distributions
function [overlap2] = calc_overlap_twonormal(s1,s2,mu1,mu2,xstart,xend,xinterval)
x_range=xstart:xinterval:xend;
overlap=cumtrapz(x_range,min([normpdf(x_range,mu1,s1)' normpdf(x_range,mu2,s2)']'));%,[],2
overlap2 = overlap(end);
end

% Function that calculates the best division for a specific level of the
% hierarchy
function[valid,C_best]=get_best_division(X_train,cidx,K_exp)
    eva = evalclusters(X_train,cidx,'DaviesBouldin');
    condInitial = eva.CriterionValues;
    C = dec2bin(0:2^K_exp-1) - '0';
    C = C(2:end-1,:);
    ovl=zeros(size(X_train,2),size(C,1),1);
    for i=1:size(C,1)
        subset1=find(C(i,:));
        % Obtain mean and std for each group of behaviors
        feat_intra_mus=mean(X_train(ismember(cidx,subset1),:),1);
        feat_intra_stds=std(X_train(ismember(cidx,subset1),:),1);
        subset2=find(C(i,:)==0);
        feat_inter_mus=mean(X_train(ismember(cidx,subset2),:),1);
        feat_inter_stds=std(X_train(ismember(cidx,subset2),:),1);
        [mu1,p1]=min([feat_intra_mus; feat_inter_mus],[],1);
        [mu2,p2]=max([feat_intra_mus; feat_inter_mus],[],1);
        aux1=[feat_intra_stds; feat_inter_stds];
        aux2=[feat_intra_stds; feat_inter_stds];
        sigma1=zeros(size(mu1));
        sigma2=zeros(size(mu2));
        for m=1:size(sigma1,2)
            sigma1(m)=aux1(p1(m),m);
            sigma2(m)=aux2(p2(m),m);
        end
        % Numerical OVL
        Npoints=10000;
        for k=1:size(ovl,1)
            xmin=min([mu1(k)-6*sigma1(k),mu2(k)-3*sigma2(k)]);
            xmax=max([mu1(k)+6*sigma1(k),mu2(k)+3*sigma2(k)]);
            ovl(k,i)= calc_overlap_twonormal(sigma1(k),sigma2(k),mu1(k),mu2(k),xmin,xmax,(xmax-xmin)/(Npoints+1));
            if (isnan(ovl(k,i)) )
                % Sigmas son cero
                if (mu1(k)==mu2(k))
                    ovl(k,i)=1;
                else
                    ovl(k,i)=0;
                end
            end
        end
    end 
    % Criterion: minimize the median of the OVLs for each group of
    % behaviors (in the division with the lesser median OVL the groups of
    % behaviors are more separable). Apart from that, get the DB criterion
    % for each division
    crit_ovl=zeros(size(C,1),1);
    davies_bouldin=zeros(size(C,1),1);
    for i=1:size(C,1)
        subset1=find(C(i,:));
        crit_ovl(i)=median(ovl(:,i));
        clust=double(ismember(cidx,subset1));
        clust(clust==0)=2;
        eva = evalclusters(X_train,clust,'DaviesBouldin');
        davies_bouldin(i)=eva.CriterionValues;
    end
    [~,p]=min(crit_ovl);
    C_best=C(p,:);
    subset1=find(C_best);
    clust=double(ismember(cidx,subset1));
    clust(clust==0)=2;
    eva = evalclusters(X_train,clust,'DaviesBouldin');
    cond = eva.CriterionValues;
    % If the DB criterion has improved
    if (cond<condInitial)
        valid=true;
    else
        valid=false;
    end
end

% Function that classifies the samples with a linear discriminant
function[error]=classifySamples(XT,yT,Xt,yt,criterion)
    error=sum(yt~=classify(Xt,XT,yT,criterion))/length(yt);
end

% Function that gets M cross-validation indexes
function [idxTrain,idxVal]=getCrossVal(N_total,M)
    h=randperm(N_total);
    div=1/M;
    folds=cell(M,1);
    for i=1:M
        folds{i}=h(round((i-1)*div*N_total+1):round(i*div*N_total));
    end
    d=zeros(M,1);
    d(1)=1;
    idxTrain=cell(M,1);
    idxVal=cell(M,1);
    for i=1:M
        d1=circshift(d,i-1);
        indexesVal=find(d1==1);
        indexesTrain=find(d1==0);
        idxTrain{i}=[];
        idxVal{i}=[];
        for j=1:length(indexesTrain)
            idxTrain{i}=[idxTrain{i} folds{indexesTrain(j)}];
        end
        for j=1:length(indexesVal)
            idxVal{i}=[idxVal{i} folds{indexesVal(j)}];
        end
    end
end