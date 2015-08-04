%function [neuron_num]=FC_RWBS_Example4%-------------------------FC_RWBS_OLS RBF网络------------------------%
    clear all;
    clc;
    load RBF_train_input;
    load RBF_train_output;

    input=RBF_train_input;
    N=size(RBF_train_input,2);
    input_sum=0;
    for num_sample_i=1:size(RBF_train_input,2)
        for num_sample_j=1:size(RBF_train_input,2)
            substrct=input(:,num_sample_i)-input(:,num_sample_j);
            distance2=sum((abs(substrct).^2));
            input_sum=input_sum+distance2;
        end
    end
    determine_gama_b=(1/2)*sqrt(input_sum/(N*(N-1)));
    gama_b=determine_gama_b*1.5;
    origin_SamInputData=RBF_train_input(:,1:size(RBF_train_input,2));
   
    jieshu=size(origin_SamInputData,1);
    gama_a=gama_b/1.5;%0.74 0.9  0.78
    Ps=5;
    Nb=10;%5;%待定
    kethB=0.005;
    keth=0.005;%判断程序结束的条件（小于keth）
    NumSam=size(origin_SamInputData,2);%为样本总数目
    Ng=20;
    D_k=zeros(NumSam,1);%与选择的互相正交的每个中心向量所对应
    for k=1:NumSam  %k不会取到NumSam,在sumJ＜keth时就停止了此大循环   
        %----------------------------------------RWBS_OLS---------------------------------------------%
        jishu1=NumSam;%%%%%%%%%%%%%%%%%%%%%%%%%%1avail_NumSam
        %%%%%%%%%%%%%%%%%%%%%%%%%%1avail_NumSam
        for n=1:Ng
        %----------------------------------------遍历随机产生中心------------------------------------------%
            u=zeros(jieshu,Ps+2);  
            delta(:,1)=(1/Ps)*ones(Ps,1); 
            ordercenter=[];
            if n==1
                RandomNumSam=randperm(jishu1);
                Number=RandomNumSam(1:Ps);
                ordercenter=origin_SamInputData(:,Number);
            else
                RandomNumSam=randperm(jishu1);
                Number=RandomNumSam(1:Ps-1);
                ordercenter=origin_SamInputData(:,Number);
            end    
        %---------------------------------------选择中心------------------------------------------------%
            if n==1%%%%%%%%%%%%%%%%%%%%%%%%%%3
                u(:,1:Ps)=ordercenter(:,1:Ps);
            else
                u(:,1)=u_best(:,n-1);
                u(:,2:Ps)=ordercenter(:,1:Ps-1);
            end
        %---------------------------------------产生Ps个中心所对应的密度指标向量----------------------------------------------%
            D=[];
            for icenter= 1:Ps
                for n_p=1:NumSam
                    substract=origin_SamInputData(1:jieshu,n_p)-u(1:jieshu,icenter);
                    Density(n_p)=exp(-((substract'*substract)/(gama_a)^2));
                end
                D(icenter,1)=sum(Density);
                if k>1
                    for m_D=1:k-1
                        D(icenter,1)=D(icenter,1)-D_k(m_D)*exp(-(u(1:jieshu,icenter)-u_ols(1:jieshu,m_D))'*(u(1:jieshu,icenter)-u_ols(1:jieshu,m_D))/(gama_b^2));
                    end
                end
            end
     %---------------------------------------WBS-------------------------------------------%
            for t=1:Nb
                u(:,Ps+1)=0;%改：添加的语句，每次t循环u(:,Ps+1)要清零
                [valueJ_worst,i_worst]=min(D);%u_worst(:,n)=u(:,i_worst);%改3：best改为u_worst
                [valueJ_best,i_best]=max(D);u_best(:,n)=u(:,i_best); %\
                normlizeJ=D/sum(D);
                if 1==t
                    eta(t)=delta(:,1)'*normlizeJ;%if 1==t就将delta(:,t)即delta(:,1)覆盖掉，否则就生成新的delta(:,t)，t=2,3...Nb来保存delta的值
                else 
                    eta(t)=delta(:,t-1)'*normlizeJ;
                end
                beta(t)=eta(t)/(1-eta(t));%计算beta
                if beta(t)<=1%改5：<= 改为 >??????????没改
                    if t==1
                        delta(:,t)=delta(:,1).*(beta(t).^(ones(Ps,1)-normlizeJ));%if 1==t就将delta(:,t)即delta(:,1)覆盖掉，否则就生成新的delta(:,t)，t=2,3...Nb来保存delta的值
                    else 
                        delta(:,t)=delta(:,t-1).*(beta(t).^(ones(Ps,1)-normlizeJ));
                    end
                else  %beta(t)>1
                    if t==1
                        delta(:,t)=delta(:,1).*(beta(t).^normlizeJ);%if 1==t就将delta(:,t)即delta(:,1)覆盖掉，否则就生成新的delta(:,t)，t=2,3...Nb来保存delta的值
                    else 
                        delta(:,t)=delta(:,t-1).*(beta(t).^normlizeJ);%
                    end 
                end
                delta(:,t)=delta(:,t)/sum(delta(:,t));%将delta归一化
% %                 martix=repmat(delta(:,t)',jieshu,1);%将delta的转置扩展成为一个(m-1)*Ps的矩阵——————step2 ??????repmat(delta(:,t)',m,1)??????
% %                 theta_u_Ps1=0;
% %                 segma_Ps1=0;
% %                 %-----------------------------------------只要改变相位---------------------------------------%
% %                 for nummartix=1:Ps              %求u(:Ps+1)
% %                     theta_u_Ps1=theta_u_Ps1+angle(u(1:jieshu,nummartix)).*martix(:,nummartix);
% %                 end
% %                 u(1:jieshu,Ps+1)=exp(1i*theta_u_Ps1);
% %                 theta_u_Ps2=angle(u_best(1:jieshu,n))+(angle(u_best(1:jieshu,n))-angle(u(1:jieshu,Ps+1))); %求u(:Ps+2)
% %                 u(1:jieshu,Ps+2)=exp(1i*theta_u_Ps2);
% %                 u(jieshu+1,Ps+1)=u_best(jieshu+1,n);
% %                 u(jieshu+1,Ps+2)=u_best(jieshu+1,n);
                u(1:jieshu,Ps+1)=u(:,1:Ps)*delta(:,t);
                u(1:jieshu,Ps+2)=(u_best(1:jieshu,n))+(u_best(1:jieshu,n)-u(1:jieshu,Ps+1));
                 %----------求u（：,Ps+1）和u（：,Ps+2）的回归量，K==1所以不需要正交，直接求theta与误差J
                for i_Ps=Ps+1:Ps+2
                    for num_input=1:NumSam
                        substract=origin_SamInputData(1:jieshu,num_input)-u(1:jieshu,i_Ps);
                        Density_Ps(num_input)=exp(-((substract'*substract)/(gama_a)^2));
                    end 
                    D_Ps(i_Ps-Ps,1)=sum(Density_Ps);
                    if k>1
                        for m_D=1:k-1
                            D_Ps(i_Ps-Ps,1)=D_Ps(i_Ps-Ps,1)-D_k(m_D)*exp(-(u(1:jieshu,i_Ps)-u_ols(1:jieshu,m_D))'*(u(1:jieshu,i_Ps)-u_ols(1:jieshu,m_D))/(gama_b^2));
                        end
                    end
                end
                 %-------------------判断J(Ps+1,k)与J(Ps+2,k)哪个小，小的用i_star表示,k==1的情况--------------------%
                if (D_Ps(1))>D_Ps(2) && (D_Ps(1))>valueJ_worst%k==1%改8：<=改为 >
                    i_star=Ps+1;
                    u(:,i_worst)=u(:,i_star);
                    D(i_worst)=D_Ps(i_star-Ps);
                end
                if (D_Ps(2))>D_Ps(1) &&(D_Ps(2))>valueJ_worst%k==1%改8：<=改为 >
                    i_star=Ps+2;
                    u(:,i_worst)=u(:,i_star);
                    D(i_worst)=D_Ps(i_star-Ps);
                end
                %-------------------判断J(Ps+1,k)与J(Ps+2,k)哪个小，小的用i_star表示,k>1的情况--------------------%
                if sum(abs(u(1:jieshu,Ps+1)-u(1:jieshu,Ps+2)))<kethB || max(D)<=0  %abs改为求角度angle的逼近(abs(theta_u_Ps1-(theta_u_Ps2+flag_above_p2*2*pi+flag_below_p2*(-2*pi))))
                    break;%跳出t=1:Nb这个循环
                end
            end   %与t=1:Nb循环对应
        end %与for n=1:NG对应                              
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if valueJ_best>0
            D_k(k)=valueJ_best;
            u_ols(:,k)=u_best(:,n);%此时n=NG 
       end
        if D_k(k)/D_k(1)<keth  
            break;%跳出k=1:NumSam循环，即整个选择中心程序结束
        end              
    end%与k=1:NumSam对
    neuron_num=size(u_ols,2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%伪逆计算权值%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    UnitCenters=u_ols;%获得被选择的中心
    %%%%根据聚类中心之间的距离求得每个中心的宽度
     for num_spread=1:size(UnitCenters,2)%size(u_ols,2)为中心的个数
        for dist_num_spread=1:size(UnitCenters,2)%计算第num_spread个中心与其他中心的距离
            if num_spread==dist_num_spread
                distance_centers(num_spread,dist_num_spread)=100;
            else
                distance_centers(num_spread,dist_num_spread)=sum(abs(UnitCenters(:,num_spread)-UnitCenters(:,dist_num_spread)));
            end
        end
        spread_cluster(1,num_spread)=min(distance_centers(num_spread,:));%得到与第num_spread个中心距离最近的中心，并计算它们的距离，将作为第num_spread个中心的宽度
     end
    for i_o=1:size(UnitCenters,2)
        for n=1:size(RBF_train_input,2)
          zt0=RBF_train_input(1:jieshu,n)-UnitCenters(1:jieshu,i_o); 
           pre_P_ols(n,i_o)=exp(-real(zt0)'*real(zt0)/(2*spread_cluster(1,i_o)^2))+1j*exp(-imag(zt0)'*imag(zt0)/(2*spread_cluster(1,i_o)^2));
        end
    end
    HiddenUnitOutSelected=pre_P_ols;%获得被选择的回归量
    HiddenUnitOutEx = [HiddenUnitOutSelected ones(size(RBF_train_input,2),1)];
    W2Ex =pinv(HiddenUnitOutEx)*RBF_train_output.'; %y需共轭转置再非共轭转置，因为在sample中其已被共轭转置过 用广义逆求广义输出初始权值矩阵   
    NNOut=HiddenUnitOutSelected*W2Ex(1:size(UnitCenters,2),:)+W2Ex(size(UnitCenters,2)+1,1);    
    figure(1)
    plot(real(NNOut),imag(NNOut),'.');
    xlabel('Real');ylabel('Imag');
    Error=RBF_train_output.'-NNOut;
    sumError_train=sum(abs(Error));    
    
     TestInputData=RBF_test_input;
    y_test=RBF_test_output;
     for i_o=1:size(UnitCenters,2)
        for n=1:size(TestInputData,2)
          zt0=TestInputData(1:jieshu,n)-UnitCenters(1:jieshu,i_o); 
          P0(i_o,n)=exp(-real(zt0)'*real(zt0)/(2*spread_cluster(1,i_o)^2))+1j*exp(-imag(zt0)'*imag(zt0)/(2*spread_cluster(1,i_o)^2));
        end
    end
    TestHiddenUnitOut=P0;
    TestNNOut=W2Ex(1:size(UnitCenters,2),:).'*TestHiddenUnitOut+W2Ex(size(UnitCenters,2)+1,1);
    %实现Decision Device功能，判断传输的是哪个符号码
    for n_decision=1:size(TestNNOut,2)
        for n_class=1:4
            distance(n_class,1)=abs(TestNNOut(1,n_decision)-S(n_class,1));
        end
        [valuemin,indexmin]=min(distance);
        classify_TestNNOut(n_decision,1)=S(indexmin,1);
    end
    %计算误码率误码率（SER：symbol error rate）是衡量数据在规定时间内数据传输精确性的指标。
    %误码率=传输中的误码/所传输的总码数*100%
    err_symbol=0;
    for n_decision=1:size(TestNNOut,2)
        if abs(classify_TestNNOut(n_decision,1)-y_test(1,n_decision))~=0
            err_symbol=err_symbol+1;
        end
    end
    err_symbol
    %计算误码率
    ser=err_symbol/(size(TestNNOut,2))
    
    save spread_cluster.mat;%计算出的宽度
    save u_ols.mat;%RWBS选择的中心
    save W2Ex.mat;%保存算出的权值
%end