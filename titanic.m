function titanic()

    set(0,'DefaultFigureWindowStyle','docked')
    clear
    clc
    fclose all;
    close all;
    rng('default');
        
    train_data=readtable('train.csv');
    test_data=readtable('test.csv');
    
    ii=find(strcmp(train_data.Embarked,''));
    train_data(ii,:)=[];

    x_train=prepare_X_data(train_data);
    x_test=prepare_X_data(test_data);
    y_train=train_data.Survived;
    
    disp('');

    K=size(unique(y_train),1);
    numHidden=6;
    eta=0.008;
    numEpochs=1000;

    [loss_train,accuracy_train,Y_Pred]=solve_NN(x_train,x_test,y_train,numHidden,numEpochs,eta,K);

    disp('');

    figure()
    subplot(1,2,1)
    plot(1:numEpochs,loss_train);
    grid on
    subplot(1,2,2)
    plot(1:numEpochs,accuracy_train);
    grid on
    
    fprintf('accuracy: %4.2f%%\n',accuracy_train(end));
    fprintf('loss: %4.4f\n',loss_train(end));

    Survived=Y_Pred;
    PassengerId=test_data{:,1};
    submission=table(PassengerId,Survived);
    
    writetable(submission,'submission.csv');
    disp('');
    
    
end





function [loss_train,accuracy_train,Y_Pred]=solve_NN(X_Train,X_Test,Y_Train,numHidden,numEpochs,eta,K)
    
    disp('');
    x_train=X_Train;
    x_test=X_Test;
    y_train=Y_Train;
    
    xmean=mean(x_train);
    xstd=std(x_train);
    x_train=(x_train-ones(size(x_train,1),1)*xmean)./(ones(size(x_train,1),1)*xstd);
    x_train=[ones(size(x_train,1),1) x_train];
    y_train_cat=(y_train==[0:(K-1)]);
    
    x_test=(x_test-ones(size(x_test,1),1)*xmean)./(ones(size(x_test,1),1)*xstd);
    x_test=[ones(size(x_test,1),1) x_test];
    
    numPatterns=size(x_train,1);

    Wh=0.1*randn(size(x_train,2),numHidden);
    wo=0.1*randn(numHidden+1,K);

    disp('');
    loss_train=zeros(numEpochs,1);
    accuracy_train=zeros(numEpochs,1);
    
    hh=waitbar(0,'');
    set(hh,'position',[-1954 595 270 56]);
    for i=1:numEpochs
        zh_train = x_train*Wh;
        phi_train=[ones(size(x_train,1),1) relu(zh_train)];
        s_train = exp(phi_train*wo);
        y_hat_train_cat = s_train./sum(s_train,2);
        loss_train(i,1)= -mean(sum((log(y_hat_train_cat)).* y_train_cat,2));
        [~, y_hat_train] = max(y_hat_train_cat,[],2);
        y_hat_train=y_hat_train-1;
        accuracy_train(i,1)=100*mean(y_hat_train==y_train);
        err_train=y_hat_train_cat - y_train_cat;
        dWh = x_train'*((err_train*wo(2:numHidden+1,:)').*(zh_train>0));
        dwo = phi_train'*err_train;
        Wh=Wh-eta/numPatterns*dWh;
        wo=wo-eta/numPatterns*dwo;
        waitbar(i/numEpochs,hh);
    end
    close(hh);
    disp('');
    
    zh_test = x_test*Wh;
    phi_test=[ones(size(x_test,1),1) relu(zh_test)];
    s_test = exp(phi_test*wo);
    Y_Pred_cat = s_test./sum(s_test,2);
    [~,Y_Pred] = max(Y_Pred_cat,[],2);
    Y_Pred=Y_Pred-1;
    
    disp('');
    
end

function x=relu(x)

    x(x<0)=0;

end


function [variable_cat]=build_cat_variable(variable)

    disp('');
    if isnumeric(variable)
        categorias=unique(variable);
        for cont=0:(size(categorias,1)-1)
            ii=find(variable==categorias(cont+1));
            variable(ii)=cont;
        end
        categorias=unique(variable);
    else
        variable_cell=variable;
        variable=zeros(size(variable_cell,1),1);
        categorias=unique(variable_cell);
        for cont=0:(size(categorias,1)-1)
            ii=find(strcmp(variable_cell,categorias(cont+1)));
            variable(ii)=cont;
        end
        categorias=unique(variable);
    end
    variable_cat=(variable==[0:(size(categorias,1)-1)]);
    
end



function X_data=prepare_X_data(X_data)

    disp('');
    [num_rows,num_cols]=size(X_data);
    
    pclass=X_data.Pclass;
    name=X_data.Name;
    fare=X_data.Fare;
    embarked=X_data.Embarked;
      
    for cont=1:num_rows
       this_name=name{cont};
       if ~isempty(strfind(this_name,'Mrs.'))
            prefix{cont,1}='Mrs';
       elseif ~isempty(strfind(this_name,'Ms.'))
            prefix{cont,1}='Mrs';
       elseif ~isempty(strfind(this_name,'Mrs'))
            prefix{cont,1}='Mrs';
       elseif ~isempty(strfind(this_name,'Dona.'))
            prefix{cont,1}='Mrs';
       elseif ~isempty(strfind(this_name,'Miss.'))
            prefix{cont,1}='Miss';      
       elseif ~isempty(strfind(this_name,'Mr.'))
            prefix{cont,1}='Mr';
       elseif ~isempty(strfind(this_name,'Don.'))
            prefix{cont,1}='Mr';
       elseif ~isempty(strfind(this_name,'Master'))
            prefix{cont,1}='Master';   
       elseif ~isempty(strfind(this_name,'Dr.'))
            prefix{cont,1}='Dr';                
       elseif ~isempty(strfind(this_name,'Rev.'))
            prefix{cont,1}='Rev';      
       elseif ~isempty(strfind(this_name,'Mme.'))
            prefix{cont,1}='Mrs';    
       elseif ~isempty(strfind(this_name,'Mlle.'))
            prefix{cont,1}='Mrs';    
       elseif ~isempty(strfind(this_name,'Countess.'))
            prefix{cont,1}='Mrs';   
       elseif ~isempty(strfind(this_name,'Col.'))
            prefix{cont,1}='Officer';   
       elseif ~isempty(strfind(this_name,'Major.'))
            prefix{cont,1}='Officer'; 
       elseif ~isempty(strfind(this_name,'Sir.'))
            prefix{cont,1}='Mr';        
       elseif ~isempty(strfind(this_name,'Capt.'))
            prefix{cont,1}='Dr';        
       elseif ~isempty(strfind(this_name,'Jonkheer.'))
            prefix{cont,1}='Dr';       
       else
           disp();
       end
    end
    
    [pclass_cat]=build_cat_variable(pclass); 
    [embarked_cat]=build_cat_variable(embarked);
    [prefix_cat]=build_cat_variable(prefix);

    length_name=zeros(size(name,1),1);
    for cont=1:size(name,1)
        length_name(cont,1)=length(name{cont});
    end
    X_data=[length_name,fare,pclass_cat,embarked_cat,prefix_cat];
    
    
end

