clear all;
clc;

% 4QAM
S=[0.7+0.7j;-0.7+0.7j;-0.7-0.7j;0.7-0.7j];

load input_output;
load chosen_centers;

order=size(RBF_train_input,1);
Max_train_times=5000;
sumerror_train=zeros(1,Max_train_times);
keth=200;
num_hidden_neuron=size(u_centers,2);
fprintf('center number = %d\n',num_hidden_neuron);
alpha_w = 0.001;
alpha_center = 0.001;
alpha_spread = 0.001;
boost = 1.01;
decay = 0.7;

% W=randn(size(RBF_train_output,1),num_hidden_neuron)+1j*randn(size(RBF_train_output,1),num_hidden_neuron);
% spread=rand(order,num_hidden_neuron)+1j*rand(order,num_hidden_neuron);

load W_Spread;
O_hidden_out=zeros(size(u_centers,2),size(RBF_train_input,2));
Y=zeros(size(RBF_train_output,1),size(RBF_train_output,2));
error=zeros(size(RBF_train_output,1),size(RBF_train_output,2));%zeros(size(RBF_train_output,1),size(RBF_train_output,2))

delta_W=zeros(size(RBF_train_output,1),size(u_centers,2));
delta_center=zeros(order,size(u_centers,2));
delta_spread=zeros(order,size(u_centers,2));
for train_times=1:Max_train_times
    record_index=[];
    
    for num_sample=1:size(RBF_train_input,2)
        for num_hidden=1:size(u_centers,2)
            zt0=RBF_train_input(1:order,num_sample)-u_centers(1:order,num_hidden);
            x=spread(:,num_hidden).'*zt0;                                                                                                                                                                                                                 % mod(2.99*pi,pi)=3.1102  
                                                                                                                                                                                                                                                          % mod(2.99*pi,pi/2)=1.5394
            if  abs(real(x)-0)<0.01 && abs(mod(imag(x),pi)-(pi/2)) < 0.05*pi% the third condition is used to prevent the case: mod(2.49*pi,pi/2)=1.5394
                x=(sign(real(x))*0.1+real(x))+(sign(imag(x))*0.1+imag(x))*1i;                    
            end

%             exp_zt0=exp(x);%e^(v'*(x-c_i))
%             exp_negtv_zt0=exp(-x);%e^(-v'*(x-c_i))
%             O_hidden_out(num_hidden,num_sample)=2/(exp_zt0+exp_negtv_zt0);
            
            O_hidden_out(num_hidden,num_sample) = sech(x);
            if O_hidden_out(num_hidden,num_sample) == 0
                fprintf('there is zero in hidden output calculation\n');
            end
        end
        
        for num_layer_output=1:size(RBF_train_output,1)
            Y(num_layer_output,num_sample)=W(num_layer_output,:)*O_hidden_out(:,num_sample);
        end
        error(:,num_sample)=RBF_train_output(:,num_sample)-Y(:,num_sample);
       
        error_delta(:,num_sample)=error(:,num_sample);
        for num_w_one=1:size(W,1)
            for num_w_two=1:size(W,2)
                delta_W(num_w_one,num_w_two)=conj(O_hidden_out(num_w_two,num_sample))*(error_delta(num_w_one,num_sample));

                substract=RBF_train_input(1:order,num_sample)-u_centers(1:order,num_w_two);%%x-c_i
%                 x=real(spread(:,num_w_two).'*substract);%real[v'*(x-c_i)]
%                 y=imag(spread(:,num_w_two).'*substract);%imag[v'*(x-c_i)]                    
%                 sech_derivative=-(2*(exp(x + y*1i) - 1/exp(x + y*1i)))/(exp(x + y*1i) + 1/exp(x + y*1i))^2;%sech(x+iy)¶ÔxÇóÆ«µ¼
% %                 
                z = spread(:,num_w_two).'* substract;
                sech_derivative = -sech(z)*tanh(z);
                 delta_spread(:,num_w_two)=error_delta(num_w_one,num_sample)*conj(W(num_w_one,num_w_two))*conj(sech_derivative)*conj(substract);
               
                 delta_center(:,num_w_two)=-error_delta(num_w_one,num_sample)*conj(W(num_w_one,num_w_two))*conj(sech_derivative)*conj(spread(:,num_w_two));
            
            end
        end
        W=W+alpha_w*delta_W;%alpha_w*delta_W multiply the learning rate here
        spread=spread+alpha_spread*delta_spread;
        u_centers=u_centers+alpha_center*delta_center;
        
        delta_W=zeros(size(RBF_train_output,1),size(u_centers,2));
        delta_center=zeros(order,size(u_centers,2));
        delta_spread=zeros(order,size(u_centers,2));
        
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
        if sumerror_train(1,train_times)>( sumerror_train(1,train_times-1) + 1 )
%             optimum_delta_w(1,train_times+1)=decay*optimum_delta_w(1,train_times);
%             optimum_delta_spread(1,train_times+1)=decay*optimum_delta_spread(1,train_times);
%             optimum_delta_center(1,train_times+1)=decay*optimum_delta_center(1,train_times);
            alpha_w=decay*alpha_w;
            alpha_spread=decay*alpha_spread;
            alpha_center=decay*alpha_center;
        elseif sumerror_train(1,train_times)<=( sumerror_train(1,train_times-1)-1 )
%             optimum_delta_w(1,train_times+1)=boost*optimum_delta_w(1,train_times);
%             optimum_delta_spread(1,train_times+1)=boost*optimum_delta_spread(1,train_times);
%             optimum_delta_center(1,train_times+1)=boost*optimum_delta_center(1,train_times);
             alpha_w=boost*alpha_w;
            alpha_spread=boost*alpha_spread;
            alpha_center=boost*alpha_center;
        else
            alpha_w=1*alpha_w;
            alpha_spread=1*alpha_spread;
            alpha_center=1*alpha_center;
        end
    end
  
end
plot(real(Y),imag(Y),'.');

TestInputData=RBF_test_input;
y_test=RBF_test_output;
for i_o=1:size(u_centers,2)
    for n=1:size(TestInputData,2)
        zt0=TestInputData(1:order,n)-u_centers(1:order,i_o);
        TestHiddenUnitOut(i_o,n) = sech(spread(:,i_o).'*zt0);
        
%         exp_zt0=exp(spread(:,i_o).'*zt0);%e^(v'*(x-c_i))
%         exp_negtv_zt0=exp(-spread(:,i_o).'*zt0);%e^(-v'*(x-c_i))
%         TestHiddenUnitOut(i_o,n)=2/(exp_zt0+exp_negtv_zt0);
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
