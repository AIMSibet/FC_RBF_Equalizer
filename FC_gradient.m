clear all;
clc;
%Example3:y_n=o_n+0.1*o_n^2+0.05*o_n^3+v_n; v_n~train_num(0,0.01)
%o_n=(0.34-j0.27)s_n+(0.87+j0.43)s_(n-1)+(0.34-j0.21)s_(n-2)

% 4QAM
S=[0.7+0.7j;-0.7+0.7j;-0.7-0.7j;0.7-0.7j];

% train_num=5000;
% num_sample=train_num*(1+20);%train_num samples for training and train_num*20 for test

load input_output;

% uniform_distribution_real=unifrnd(-1,1,1, num_sample);
% uniform_distribution_imag=unifrnd(-1,1,1, num_sample);
% for num=1:num_sample
% 
%     if uniform_distribution_real(1,num)>=0
%         uniform_distribution_real(1,num)=0.7;
%     else
%         uniform_distribution_real(1,num)=-0.7;
%     end
% 
%     if uniform_distribution_imag(1,num)>=0
%         uniform_distribution_imag(1,num)=0.7;
%     else
%         uniform_distribution_imag(1,num)=-0.7;
%     end
% end
% 
% origin_s=uniform_distribution_real+1i*uniform_distribution_imag;
% 
% clear uniform_distribution_real uniform_distribution_imag;
% dimension_num=3;
% input_sequence=zeros(dimension_num,num_sample);
% for num=1:num_sample
%     for dimension=1:3
%         if num==1 %vector s(n) = [s(n) ... s(n-m+2-nh)] where m is the equalizer dimension and nh is the equalizer decision delay, here m=3 and nh = 3 
%             if dimension==1
%                 input_sequence(dimension,num)=origin_s(1,num);
%             elseif dimension==2
%                 input_sequence(dimension,num)=0.7-0.7j;
%             else
%                 input_sequence(dimension,num)=0.7+0.7j;
%             end
%         elseif num==2
%             if dimension==1
%                 input_sequence(dimension,num)=origin_s(1,num);
%             elseif dimension==2
%                 input_sequence(dimension,num)= origin_s(1,num-1);
%             else
%                 input_sequence(dimension,num)=0.7-0.7j;
%             end
%         else
%             if dimension==1
%                 input_sequence(dimension,num)=origin_s(1,num);
%             elseif dimension==2
%                 input_sequence(dimension,num)=origin_s(1,num-1);
%             else
%                 input_sequence(dimension,num)=origin_s(1,num-2);
%             end
%         end
%     end
% end
% 
% o_n=[0.34-0.27j 0.87+0.43j 0.34-0.21j]*input_sequence;
% clear origin_s;
% 
% bar_y=o_n+0.1*o_n.^2+0.05*o_n.^3;
% clear o_n;
% 
% y=awgn(bar_y,16,'measured','dB');
% clear bar_y;
% 
% for num=1:num_sample
%     for dimension=1:3
%         if num==1
%             if dimension==1
%                 equalizer_input(dimension,num)=y(1,num);
%             elseif dimension==2
%                 equalizer_input(dimension,num)=y(1,num);
%             else
%                 equalizer_input(dimension,num)=y(1,num);
%             end
%         elseif num==2
%             if dimension==1
%                 equalizer_input(dimension,num)=y(1,num);
%             elseif dimension==2
%                 equalizer_input(dimension,num)=y(1,num-1);
%             else
%                 equalizer_input(dimension,num)=y(1,num-1);
%             end
%         else
%             if dimension==1
%                 equalizer_input(dimension,num)=y(1,num);
%             elseif dimension==2
%                 equalizer_input(dimension,num)=y(1,num-1);
%             else
%                 equalizer_input(dimension,num)=y(1,num-2);
%             end
%         end
%     end
% end
% 
% % %dealy=1
% % equalizer_out = input_sequence(2,1:num_sample);
% 
% %delay = 3
% equalizer_out = zeros(1,num_sample);
% equalizer_out(1) = 0.7+0.7j;
% equalizer_out(2:num_sample) = input_sequence(1,1:num_sample-1);
% 
% clear input_sequence;
% 
% RBF_train_input=equalizer_input(:,1:train_num);
% RBF_train_output=equalizer_out(1,1:train_num);
% 
% RBF_test_input=equalizer_input(:,train_num+1:num_sample);
% RBF_test_output=equalizer_out(1,train_num+1:num_sample);
% 
% save input_output RBF_train_input RBF_train_output RBF_test_input RBF_test_output;
% 
% clear equalizer_input equalizer_out y;
% 
% %*************************************************RWBS_CE***************%
% input_sum=0;
% for num_sample_i=1:train_num
%     for num_sample_j=1:train_num
%         substrct=RBF_train_input(:,num_sample_i)-RBF_train_input(:,num_sample_j);
%         distance2=sum(( (abs(substrct) ).^2));
%         input_sum=input_sum+distance2;
%     end
% end
% gama_a=sqrt(input_sum/(train_num*(train_num-1)));
% gama_b=gama_a*1.5;

%-----------define global variables and initialization---------------%
order=size(RBF_train_input,1);

% Ps=5; % the population size
% Nb=10;%5; % the maximum iterations in the WBS process, need adjustment
% kethB=0.005; % a given positive scalar in the termination criterion of the WBS process
% keth=0.005;%stopping criterion (smaller keth)
% Ng=20; %the maximum iterations in the RWBS process
% D_k=zeros(train_num,1);%
% %----------------------------------------RWBS_CE---------------------------------------------%
% for k=1:train_num
%     
%     %----------------------------------------RWBS---------------------------------------------%
%     for n=1:Ng
%         %----------------------------------------generate the preliminary centers randomly from the whole training set------------------------------------------%
%         u=zeros(order,Ps+2);
%         delta(:,1)=(1/Ps)*ones(Ps,1);
%         preliminaryCenters=[];
%         if n==1
%             Randomtrain_num=randperm(train_num);
%             Number=Randomtrain_num(1:Ps);
%             preliminaryCenters=RBF_train_input(:,Number);
%         else
%             Randomtrain_num=randperm(train_num);
%             Number=Randomtrain_num(1:Ps-1);
%             preliminaryCenters=RBF_train_input(:,Number);
%         end
%         %---------------------------------------choose the centers------------------------------------------------%
%         if n==1
%             u(:,1:Ps)=preliminaryCenters(:,1:Ps);
%         else
%             u(:,1)=u_best(:,n-1); %u_best(:,n-1) has the largest density in the previous iteration
%             u(:,2:Ps)=preliminaryCenters(:,1:Ps-1);
%         end
%         %---------------------------------------generate the cluster density corresponding to the Ps centers ----------------------------------------------%
%         D=[];
%         for icenter= 1:Ps
%             for n_p=1:train_num
%                 substract=RBF_train_input(1:order,n_p)-u(1:order,icenter);
%                 Density(n_p)=exp(-((substract'*substract)/(gama_a/2)^2));
%             end
%             D(icenter,1)=sum(Density);
%             if k>1
%                 for m_D=1:k-1%D_k(m_D) represents the density of the previous center
%                     D(icenter,1)=D(icenter,1)-D_k(m_D)*exp(-(u(1:order,icenter)-u_centers(1:order,m_D))'*(u(1:order,icenter)-u_centers(1:order,m_D))/((gama_b/2)^2));
%                 end
%             end
%         end
%         %---------------------------------------WBS-------------------------------------------%
%         for t=1:Nb
%             u(:,Ps+1)=0;%u(:,Ps+1) should be set to 0 in every iteration ???????????????????
%             [valueJ_worst,i_worst]=min(D);
%             if t==1
%                 [valueJ_best,i_best]=max(D);
%                 u_best(:,n)=u(:,i_best);
%             end
%             normlizeJ=D/sum(D);
%             if 1==t
%                 eta(t)=delta(:,1)'*normlizeJ;
%             else
%                 eta(t)=delta(:,t-1)'*normlizeJ;
%             end
%             beta(t)=eta(t)/(1-eta(t));%calculate beta
%             if beta(t)<=1
%                 if t==1
%                     delta(:,t)=delta(:,1).*(beta(t).^(ones(Ps,1)-normlizeJ));%update delta(:,1)
%                 else
%                     delta(:,t)=delta(:,t-1).*(beta(t).^(ones(Ps,1)-normlizeJ));
%                 end
%             else  %beta(t)>1
%                 if t==1
%                     delta(:,t)=delta(:,1).*(beta(t).^normlizeJ);
%                 else
%                     delta(:,t)=delta(:,t-1).*(beta(t).^normlizeJ);%
%                 end
%             end
%             delta(:,t)=delta(:,t)/sum(delta(:,t));
%             u(1:order,Ps+1)=u(:,1:Ps)*delta(:,t);
%             u(1:order,Ps+2)=(u_best(1:order,n))+(u_best(1:order,n)-u(1:order,Ps+1));
%             
%             for i_Ps=Ps+1:Ps+2
%                 for num_input=1:train_num
%                     substract=RBF_train_input(1:order,num_input)-u(1:order,i_Ps);
%                     Density_Ps(num_input)=exp(-((substract'*substract)/(gama_a/2)^2));
%                 end
%                 D_Ps(i_Ps-Ps,1)=sum(Density_Ps);
%                 if k>1
%                     for m_D=1:k-1
%                         D_Ps(i_Ps-Ps,1)=D_Ps(i_Ps-Ps,1)-D_k(m_D)*exp(-(u(1:order,i_Ps)-u_centers(1:order,m_D))'*(u(1:order,i_Ps)-u_centers(1:order,m_D))/((gama_b/2)^2));
%                     end
%                 end
%             end
%             %-------------------choose the bigger one between J(Ps+1,k) and J(Ps+2,k) as i_star--------------------%
%             if (D_Ps(1))>D_Ps(2) && (D_Ps(1))>valueJ_worst
%                 i_star=Ps+1;
%                 u(:,i_worst)=u(:,i_star); % update the worst center with the better generated data
%                 D(i_worst)=D_Ps(i_star-Ps);
%             end
%             if (D_Ps(2))>D_Ps(1) &&(D_Ps(2))>valueJ_worst
%                 i_star=Ps+2;
%                 u(:,i_worst)=u(:,i_star);
%                 D(i_worst)=D_Ps(i_star-Ps);
%             end
%             
%             %*new added*%
%             [valueJ_best,i_best]=max(D);
%             u_best(:,n)=u(:,i_best);
%             
%             if sum(abs(u(1:order,Ps+1)-u(1:order,Ps+2)))<kethB || max(D)<=0
%                 break;%break the WBS process
%             end
%         end   %the WBS process
%     end % the RWBS process
% 
%     if valueJ_best>0
%         D_k(k)=valueJ_best;
%         u_centers(:,k)=u_best(:,n);%n=train_numG
%     end
%     if D_k(k)/D_k(1)<keth
%         break;
%     end
% end%k=1:train_num
% save chosen_centers u_centers;

load chosen_centers;
num_hidden_neuron = size(u_centers,2);
fprintf('center number = %d\n',num_hidden_neuron);


jieshu=size(RBF_train_input,1);
Max_train_times=800;
sumerror_train=zeros(1,Max_train_times);
keth=200;
num_hidden_neuron=size(u_centers,2);%10;
optimum_delta_w=[0.001 0.001 0.001*ones(1,Max_train_times-2)];%0.01;%(error_w'*hidden_out*hidden_out'*error_w)/(error_w'*hidden_out*hidden_out'*hidden_out*hidden_out'*error_w);
optimum_delta_center=[0.001 0.001 0.001*ones(1,Max_train_times-2)];%0.05;
optimum_delta_spread=[0.001 0.001 0.001*ones(1,Max_train_times-2)];%0.001;


% W=randn(size(RBF_train_output,1),num_hidden_neuron)+1j*randn(size(RBF_train_output,1),num_hidden_neuron);
% spread=rand(order,num_hidden_neuron)+1j*rand(order,num_hidden_neuron);

load W_spread;

O_hidden_out=zeros(size(u_centers,2),size(RBF_train_input,2));
Y=zeros(size(RBF_train_output,1),size(RBF_train_output,2));
error=zeros(size(RBF_train_output,1),size(RBF_train_output,2));%zeros(size(RBF_train_output,1),size(RBF_train_output,2))

delta_W=zeros(size(RBF_train_output,1),size(u_centers,2));
delta_center=zeros(jieshu,size(u_centers,2));
delta_spread=zeros(jieshu,size(u_centers,2));
for train_times=1:Max_train_times
    record_index=[];
    for num_sample=1:size(RBF_train_input,2)
        %-------------------ÇóÒþ²ãÊä³ö---------------------%
        for num_hidden=1:size(u_centers,2)
            zt0=RBF_train_input(1:jieshu,num_sample)-u_centers(1:jieshu,num_hidden);
            x=spread(:,num_hidden).'*zt0;
            if  abs(real(x)-0)<0.01 && abs(mod(imag(x),pi)-(pi/2)) < 0.05*pi % the third condition is used to prevent the case: mod(2.49*pi,pi/2)=1.5394
                x=(sign(real(x))*0.1+real(x))+(sign(imag(x))*0.1+imag(x))*1i;                    
            end
            exp_zt0=exp(x);%e^(v'*(x-c_i))
            exp_negtv_zt0=exp(-x);%e^(-v'*(x-c_i))
            O_hidden_out(num_hidden,num_sample)=2/(exp_zt0+exp_negtv_zt0);
        end
   
        for num_layer_output=1:size(RBF_train_output,1)
            Y(num_layer_output,num_sample)=W(num_layer_output,:)*O_hidden_out(:,num_sample);
        end
        error(:,num_sample)=RBF_train_output(:,num_sample)-Y(:,num_sample);
        
        error_delta(:,num_sample)=error(:,num_sample);
        for num_w_one=1:size(W,1)
            for num_w_two=1:size(W,2)
  
                delta_W(num_w_one,num_w_two)=optimum_delta_w(1,train_times)*conj(O_hidden_out(num_w_two,num_sample))*(error_delta(num_w_one,num_sample));
                substract=RBF_train_input(1:jieshu,num_sample)-u_centers(1:jieshu,num_w_two);%%x-c_i
                x=real(spread(:,num_w_two).'*substract);%real[v'*(x-c_i)]
                y=imag(spread(:,num_w_two).'*substract);%imag[v'*(x-c_i)]
                temp2=-(2*(exp(x + y*1i) - 1/exp(x + y*1i)))/(exp(x + y*1i) + 1/exp(x + y*1i))^2;
     
                delta_spread(:,num_w_two)=optimum_delta_spread(1,train_times)*error_delta(num_w_one,num_sample)*conj(W(num_w_one,num_w_two))*conj(temp2)*conj(substract);
         
                delta_center(:,num_w_two)=-optimum_delta_center(1,train_times)*error_delta(num_w_one,num_sample)*conj(W(num_w_one,num_w_two))*conj(temp2)*conj(spread(:,num_w_two));
            end
        end
        W=W+delta_W;
        spread=spread+delta_spread;
        u_centers=u_centers+delta_center;
    end
    sumerror_train(1,train_times)=0.5*sum(abs(error(:)).^2);
    error_train=sumerror_train(1,train_times);
    
    if mod(train_times,50) == 0
        fprintf('current error = %f\n',error_train);
    end
     
    if error_train<keth
        break
    end
    
    if train_times>=2
        if sumerror_train(1,train_times) > (sumerror_train(1,train_times-1)+1)
            optimum_delta_w(1,train_times+1)=0.7*optimum_delta_w(1,train_times);
            optimum_delta_spread(1,train_times+1)=0.7*optimum_delta_spread(1,train_times);
            optimum_delta_center(1,train_times+1)=0.7*optimum_delta_center(1,train_times);
        elseif sumerror_train(1,train_times)<= (sumerror_train(1,train_times-1)-1)
            optimum_delta_w(1,train_times+1)=1.01*optimum_delta_w(1,train_times);
            optimum_delta_spread(1,train_times+1)=1.01*optimum_delta_spread(1,train_times);
            optimum_delta_center(1,train_times+1)=1.01*optimum_delta_center(1,train_times);
        else
            optimum_delta_w(1,train_times+1)=optimum_delta_w(1,train_times);
        end
    end
  
end
plot(real(Y),imag(Y),'.');

TestInputData=RBF_test_input;
y_test=RBF_test_output;
for i_o=1:size(u_centers,2)
    for n=1:size(TestInputData,2)
        zt0=TestInputData(1:jieshu,n)-u_centers(1:jieshu,i_o);
        exp_zt0=exp(spread(:,i_o).'*zt0);%e^(v'*(x-c_i))
        exp_negtv_zt0=exp(-spread(:,i_o).'*zt0);%e^(-v'*(x-c_i))
        TestHiddenUnitOut(i_o,n)=2/(exp_zt0+exp_negtv_zt0);
    end
end

TestNNOut=W*TestHiddenUnitOut;
for n_decision=1:size(TestNNOut,2)
    for n_class=1:4
        distance(n_class,1)=abs(TestNNOut(1,n_decision)-S(n_class,1));
    end
    [valuemin,indexmin]=min(distance);
    classify_TestNNOut(n_decision,1)=S(indexmin,1);
end

err_symbol=0;
for n_decision=1:size(TestNNOut,2)
    if abs(classify_TestNNOut(n_decision,1)-y_test(1,n_decision))~=0
        err_symbol=err_symbol+1;
    end
end

ser=err_symbol/(size(TestNNOut,2));
fprintf('FC Test SER = %f\n',ser);
