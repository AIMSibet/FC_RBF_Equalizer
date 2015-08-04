%function [neuron_num]=FC_RWBS_Example4%-------------------------FC_RWBS_OLS RBF����------------------------%
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
    Nb=10;%5;%����
    kethB=0.005;
    keth=0.005;%�жϳ��������������С��keth��
    NumSam=size(origin_SamInputData,2);%Ϊ��������Ŀ
    Ng=20;
    D_k=zeros(NumSam,1);%��ѡ��Ļ���������ÿ��������������Ӧ
    for k=1:NumSam  %k����ȡ��NumSam,��sumJ��kethʱ��ֹͣ�˴˴�ѭ��   
        %----------------------------------------RWBS_OLS---------------------------------------------%
        jishu1=NumSam;%%%%%%%%%%%%%%%%%%%%%%%%%%1avail_NumSam
        %%%%%%%%%%%%%%%%%%%%%%%%%%1avail_NumSam
        for n=1:Ng
        %----------------------------------------���������������------------------------------------------%
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
        %---------------------------------------ѡ������------------------------------------------------%
            if n==1%%%%%%%%%%%%%%%%%%%%%%%%%%3
                u(:,1:Ps)=ordercenter(:,1:Ps);
            else
                u(:,1)=u_best(:,n-1);
                u(:,2:Ps)=ordercenter(:,1:Ps-1);
            end
        %---------------------------------------����Ps����������Ӧ���ܶ�ָ������----------------------------------------------%
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
                u(:,Ps+1)=0;%�ģ���ӵ���䣬ÿ��tѭ��u(:,Ps+1)Ҫ����
                [valueJ_worst,i_worst]=min(D);%u_worst(:,n)=u(:,i_worst);%��3��best��Ϊu_worst
                [valueJ_best,i_best]=max(D);u_best(:,n)=u(:,i_best); %\
                normlizeJ=D/sum(D);
                if 1==t
                    eta(t)=delta(:,1)'*normlizeJ;%if 1==t�ͽ�delta(:,t)��delta(:,1)���ǵ�������������µ�delta(:,t)��t=2,3...Nb������delta��ֵ
                else 
                    eta(t)=delta(:,t-1)'*normlizeJ;
                end
                beta(t)=eta(t)/(1-eta(t));%����beta
                if beta(t)<=1%��5��<= ��Ϊ >??????????û��
                    if t==1
                        delta(:,t)=delta(:,1).*(beta(t).^(ones(Ps,1)-normlizeJ));%if 1==t�ͽ�delta(:,t)��delta(:,1)���ǵ�������������µ�delta(:,t)��t=2,3...Nb������delta��ֵ
                    else 
                        delta(:,t)=delta(:,t-1).*(beta(t).^(ones(Ps,1)-normlizeJ));
                    end
                else  %beta(t)>1
                    if t==1
                        delta(:,t)=delta(:,1).*(beta(t).^normlizeJ);%if 1==t�ͽ�delta(:,t)��delta(:,1)���ǵ�������������µ�delta(:,t)��t=2,3...Nb������delta��ֵ
                    else 
                        delta(:,t)=delta(:,t-1).*(beta(t).^normlizeJ);%
                    end 
                end
                delta(:,t)=delta(:,t)/sum(delta(:,t));%��delta��һ��
% %                 martix=repmat(delta(:,t)',jieshu,1);%��delta��ת����չ��Ϊһ��(m-1)*Ps�ľ��󡪡���������step2 ??????repmat(delta(:,t)',m,1)??????
% %                 theta_u_Ps1=0;
% %                 segma_Ps1=0;
% %                 %-----------------------------------------ֻҪ�ı���λ---------------------------------------%
% %                 for nummartix=1:Ps              %��u(:Ps+1)
% %                     theta_u_Ps1=theta_u_Ps1+angle(u(1:jieshu,nummartix)).*martix(:,nummartix);
% %                 end
% %                 u(1:jieshu,Ps+1)=exp(1i*theta_u_Ps1);
% %                 theta_u_Ps2=angle(u_best(1:jieshu,n))+(angle(u_best(1:jieshu,n))-angle(u(1:jieshu,Ps+1))); %��u(:Ps+2)
% %                 u(1:jieshu,Ps+2)=exp(1i*theta_u_Ps2);
% %                 u(jieshu+1,Ps+1)=u_best(jieshu+1,n);
% %                 u(jieshu+1,Ps+2)=u_best(jieshu+1,n);
                u(1:jieshu,Ps+1)=u(:,1:Ps)*delta(:,t);
                u(1:jieshu,Ps+2)=(u_best(1:jieshu,n))+(u_best(1:jieshu,n)-u(1:jieshu,Ps+1));
                 %----------��u����,Ps+1����u����,Ps+2���Ļع�����K==1���Բ���Ҫ������ֱ����theta�����J
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
                 %-------------------�ж�J(Ps+1,k)��J(Ps+2,k)�ĸ�С��С����i_star��ʾ,k==1�����--------------------%
                if (D_Ps(1))>D_Ps(2) && (D_Ps(1))>valueJ_worst%k==1%��8��<=��Ϊ >
                    i_star=Ps+1;
                    u(:,i_worst)=u(:,i_star);
                    D(i_worst)=D_Ps(i_star-Ps);
                end
                if (D_Ps(2))>D_Ps(1) &&(D_Ps(2))>valueJ_worst%k==1%��8��<=��Ϊ >
                    i_star=Ps+2;
                    u(:,i_worst)=u(:,i_star);
                    D(i_worst)=D_Ps(i_star-Ps);
                end
                %-------------------�ж�J(Ps+1,k)��J(Ps+2,k)�ĸ�С��С����i_star��ʾ,k>1�����--------------------%
                if sum(abs(u(1:jieshu,Ps+1)-u(1:jieshu,Ps+2)))<kethB || max(D)<=0  %abs��Ϊ��Ƕ�angle�ıƽ�(abs(theta_u_Ps1-(theta_u_Ps2+flag_above_p2*2*pi+flag_below_p2*(-2*pi))))
                    break;%����t=1:Nb���ѭ��
                end
            end   %��t=1:Nbѭ����Ӧ
        end %��for n=1:NG��Ӧ                              
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if valueJ_best>0
            D_k(k)=valueJ_best;
            u_ols(:,k)=u_best(:,n);%��ʱn=NG 
       end
        if D_k(k)/D_k(1)<keth  
            break;%����k=1:NumSamѭ����������ѡ�����ĳ������
        end              
    end%��k=1:NumSam��
    neuron_num=size(u_ols,2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%α�����Ȩֵ%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    UnitCenters=u_ols;%��ñ�ѡ�������
    %%%%���ݾ�������֮��ľ������ÿ�����ĵĿ��
     for num_spread=1:size(UnitCenters,2)%size(u_ols,2)Ϊ���ĵĸ���
        for dist_num_spread=1:size(UnitCenters,2)%�����num_spread���������������ĵľ���
            if num_spread==dist_num_spread
                distance_centers(num_spread,dist_num_spread)=100;
            else
                distance_centers(num_spread,dist_num_spread)=sum(abs(UnitCenters(:,num_spread)-UnitCenters(:,dist_num_spread)));
            end
        end
        spread_cluster(1,num_spread)=min(distance_centers(num_spread,:));%�õ����num_spread�����ľ�����������ģ����������ǵľ��룬����Ϊ��num_spread�����ĵĿ��
     end
    for i_o=1:size(UnitCenters,2)
        for n=1:size(RBF_train_input,2)
          zt0=RBF_train_input(1:jieshu,n)-UnitCenters(1:jieshu,i_o); 
           pre_P_ols(n,i_o)=exp(-real(zt0)'*real(zt0)/(2*spread_cluster(1,i_o)^2))+1j*exp(-imag(zt0)'*imag(zt0)/(2*spread_cluster(1,i_o)^2));
        end
    end
    HiddenUnitOutSelected=pre_P_ols;%��ñ�ѡ��Ļع���
    HiddenUnitOutEx = [HiddenUnitOutSelected ones(size(RBF_train_input,2),1)];
    W2Ex =pinv(HiddenUnitOutEx)*RBF_train_output.'; %y�蹲��ת���ٷǹ���ת�ã���Ϊ��sample�����ѱ�����ת�ù� �ù���������������ʼȨֵ����   
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
    %ʵ��Decision Device���ܣ��жϴ�������ĸ�������
    for n_decision=1:size(TestNNOut,2)
        for n_class=1:4
            distance(n_class,1)=abs(TestNNOut(1,n_decision)-S(n_class,1));
        end
        [valuemin,indexmin]=min(distance);
        classify_TestNNOut(n_decision,1)=S(indexmin,1);
    end
    %���������������ʣ�SER��symbol error rate���Ǻ��������ڹ涨ʱ�������ݴ��侫ȷ�Ե�ָ�ꡣ
    %������=�����е�����/�������������*100%
    err_symbol=0;
    for n_decision=1:size(TestNNOut,2)
        if abs(classify_TestNNOut(n_decision,1)-y_test(1,n_decision))~=0
            err_symbol=err_symbol+1;
        end
    end
    err_symbol
    %����������
    ser=err_symbol/(size(TestNNOut,2))
    
    save spread_cluster.mat;%������Ŀ��
    save u_ols.mat;%RWBSѡ�������
    save W2Ex.mat;%���������Ȩֵ
%end