clear all
close all

% A matrix of the house prices, defined by definition transaction price
prices = dlmread('Dataminus2bornholmlasoarofano.txt'); %First column is an index column
%Running shows that  at row 3 (Frederiksberg) column 65 (2007Q4) data is missing; I add the
%aritmetic mean between time-adjacent prices: 30994 + 39382 = 35188
price = prices(:,2:(length(prices)-16)); %The price matrix. Each columns is the prices for a given time.
priceall = prices(:,2:length(prices));
%plot(1:106,prices(88,2:107)) 
% %Check how the prices evolves with time
%price(:,93) = dlmread('Pred93-108BS.txt');


%Adding the Bootstrapped predicted valuesto price:
priceadd1 = dlmread('Pred93VarBS.txt');
price(:,93) = 10000*priceadd1;
priceadd2 = dlmread('Pred94VarBS.txt');
price(:,94) = 100000*priceadd2;
priceadd3 = dlmread('Pred95VarBS.txt');
price(:,95) = 100000*priceadd3;
priceadd4 = dlmread('Pred96VarBS.txt');
price(:,96) = 1000000*priceadd4;
priceadd5 = dlmread('Pred97VarBS.txt');
price(:,97) = 1000000*priceadd5;
priceadd6 = dlmread('Pred98VarBS.txt');
price(:,98) = 10000000*priceadd6;
priceadd7 = dlmread('Pred99VarBS.txt');
price(:,99) = 10000000*priceadd7;
priceadd8 = dlmread('Pred100VarBS.txt');
price(:,100) = 100000000*priceadd8;
priceadd9 = dlmread('Pred101VarBS.txt');
price(:,101) = 100000000*priceadd9;
priceadd10 = dlmread('Pred102VarBS.txt');
price(:,102) = 1000000000*priceadd10;
priceadd11 = dlmread('Pred103VarBS.txt');
price(:,103) = 1000000000*priceadd11;
priceadd12 = dlmread('Pred104VarBS.txt');
price(:,104) = 10000000000*priceadd12;
priceadd13 = dlmread('Pred105VarBS.txt');
price(:,105) = 10000000000*priceadd13;
priceadd14 = dlmread('Pred106VarBS.txt');
price(:,106) = 100000000000*priceadd14;
priceadd15 = dlmread('Pred107VarBS.txt');
price(:,107) = 100000000000*priceadd15;
priceadd16 = dlmread('Pred108VarBS.txt');
price(:,108) = 10000000000000*priceadd16;
%The equations to solve are: x(i,t+1) = a(i) + b*t + sum[c(i,j)* x(i,t))
%for all regions j, in every region i.
r = length(price(:,1)); %Number of regions 
t = 1:size(price,2); %Vector with number of time intervalls of 3 months
nt= length(t); %Number of time intervals

%Nu räknar vi på det nya sättet. Vi använder oss av Concise matrix notation
%här https://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)
% Y = RZ+error

Z = zeros(r+2,nt-1);
Z(1,:) = ones(1,nt-1);
Z(2,:) = (1:(nt-1));
Z(3:(r+2),:) = price(:,1:(nt-1));


y = price(:,2:nt);

A=y*transpose(Z);
B= Z*transpose(Z)+0.0*eye(size(Z*transpose(Z)));
R=A*pinv(B);
%R = A/B; %This matrix contains all parameters as in Wikipedia link
%B = y*transpose(Z)/(Z*transpose(Z));

a = R(:,1); %Alpha constants vector a(i)
b = R(:,2); %Beta constants vector (b) everywhere
c = R(:,3:(r+2)); %Matrix of all coupling constants c(i,j)
%c(i,j) represents the price coupling on region i due to region j


%est = x;
%---------------------CALCULATE PRICES--------------------
%Let x represent the matrix of propagated prices one time step; x(r,t) is
%the propagated price at region r and time t (the data is taken from the
%prices at t-1 acostfnd at the first time we keep 0's). First we define its size with 0's:
x = zeros(r,108);
%x(:,2) = a + b * t(1) + c*x(:,1);
x(:,1)=price(:,1);
%x(:,107)=price(:,107);
%error = zeros(r,nt-1);

%price = prices(:,2:(length(prices)));
t = 1:108;
%PREDICTIONS
s = 93; %Starting value
f = 108; % Finishing value
x(:,s-1) = price(:,s-1);
for i=s:f
    x(:,i) = a + b*t(i-1) + c*x(:,i-1);
    %costf(:,i-1) = (x(:,i)-price(:,i)).^2; %The cost function matrix (region by region) 
    %shows the squared errors
end
%error = price-priceall;
error = x-priceall;
costf = error.^2; %The sum of these elements is the cost function that is gonna be minimized
%sum(costf);
%accuracy =(costf.^(1/2))./price;
%costf = (x(:,2)-price(:,2)).^2; %The cost function vector (region by region)
%Is the average difference (as part of price) 
%between the propagated price 1 step in time and the price
s = 77; f=77;
MSE = mean(mean(costf(:,s:f)));
MAE  = mean(mean(abs(error(:,s:f))));
MAPE = 100*mean(mean(abs(error(:,s:f))./priceall(:,s:f)));
ME   = -mean(mean(error(:,s:f)));
MPE  = -100*mean(mean(error(:,s:f)./priceall(:,s:f)));

stat(1) = MSE;
stat(2) = MAE;
stat(3) = MAPE;
stat(4) = ME;
stat(5) = MPE;

%Prediction statistics

%1. How much does the prediction say we will earn
p1 = (x(:,f)-priceall(:,s-1))./priceall(:,s-1); %Vector with predicted %-change in price for every region after the period
Positive = p1 > 0.1;
p1(~Positive) = 0; %Vector with only positive predicted profits, the negative predictions are 0.
p1 = nonzeros(p1);
sort(p1);
preprofit = mean(p1) %Predicted profit by investing equally much in all regions that predicts increase in price


r1 = (priceall(:,f)-priceall(:,s-1))./priceall(:,s-1); %Real price profit in all regions during the same time
r1(~Positive)=0; %Vector with the real predicted profits
r1 = nonzeros(r1);
realprofit = mean(r1);

mesh(error(:,93:108))
xlabel('Time (year)')
ylabel('Region index')
xticklabels({'2015','2016','2017','2018','2019'})
zlabel('House price (dkk/m^2)')
title('Errors of Bootstrapped predictions in validation period')
colorbar

%costftot = sum(costf(:,2:nt), 'all');
%fel=(costftot/(r*(nt-1))).^(0.5)/avpricetot;%The average error relative to the average price

%imagesc(abs(error(:,2:nt)))
%imagesc(c)
%colorbar
%title('Price coupling parameter value visualized by color');
%xlabel('Region index')
%ylabel('Region index')


%Check matrix symmetry
csymm = 0.5*(c+c');
canti = 0.5*(c-c');

cs = (norm(csymm,'fro')-norm(canti,'fro'))/(norm(csymm,'fro')+norm(canti,'fro'));


%Real predictions
x2 = prices(:,2:(length(prices)));
%x(:,1) = price(:,1);
t = 1:(size(x2,2));
%for i = 104:107
%x2(:,i) = a + b*t(i-1) + c*x2(:,i-1);
%end

%plot(1:r,x2(:,104),'+')
%hold on
%plot(1:r,prices(:,105))

%Let's see if we made a good prediction.
change(:,1) = x2(:,107) - prices(:,107); %Predicted change
change(:,2) = prices(:,108)-prices(:,107); %Real change
change(:,3) = change(:,1)./change(:,2); %Predicted change/Real change

%plot(1:r,change(:,1),'+')
%hold on
%plot(1:r,change(:,2),'.')

%plot(1:r,change(:,3))
%legend('Predicted price change/real price change t=107')

%Hitta antal positiva (rätt håll) och negativa (fel håll) förutsägelser
indices = find(change(:,3)<0);
%length(indices)
%--------------------------------------------------------------
%error = (x2(:,104:107)-prices(:,105:108))./prices(:,105:108);%Relative error
%histfit(error(:,3))
%legend('Relative error in price, first prediction, t = 104')


%cd = diag(diag(c));
%co = c-cd;
%histfit(reshape(cd,[],1))

%c-fördelning
%histfit(reshape(c,[],1))
%legend('Distribution of price coupling parameter values')
%xlabel('Parameter value')
%ylabel('Number of parameters')
%xlim([-6 6])

%Check the other inequality in (ii)
%plot(1:r,diag(c)./mean(c)')

%Plot the prices
%mesh(prices(:,2:109))
%xlabel('Time (year)')
%ylabel('Region index') 
%xticklabels({'1992','1997','2002','2007','2012','2017','2022'})
%zlabel('Price (dkr/m^2)')
%title('The house price in all 93 regions from 1992 to end of 2018')

%Check the distances between all regions
DM = zeros(r);
populations = dlmread('populations.txt');
for k = 1:r
    for j = k:r
    DM(k,j) = lldistkm([populations(j,10) populations(j,11)],[populations(k,10) populations(k,11)]);
    DM(j,k) = DM(k,j);
    end
end
%imagesc(DM)
%colorbar
%xlabel('Region index')
%ylabel('Region index')
%zlabel('Distance/km')
%title('Distance between regions represented in km and by colour')

%Check correlation and more of distance vs coupling constant
cvector = reshape(c,[],1);
abscvector= abs(cvector);
DMvector = reshape(DM,[],1);
[firstsordered, firstsortorder] = sort(DMvector);
secondsorted = cvector(firstsortorder);

plot(firstsordered,secondsorted,'.')
xlabel('Distance between pair of regions (km)')
ylabel('Price coupling parameters')
title('Price coupling parameters versus the distance between the regions')

%Check correlation and more of population vs coupling constant
Pvector = 1000/4*(populations(:,2) + populations(:,4) + populations(:,6) + populations(:,8))+ 1/4*(populations(:,3) + populations(:,5) + populations(:,7) + populations(:,9));
[firstsordered, firstsortorder] = sort(Pvector);
secondsorted = cvector(firstsortorder);

for i = 1:r
    c1 = c(i,:);
    secondsorted = c1(firstsortorder);
    info = corrcoef(firstsordered,secondsorted);
    corr(i) = info(2,1);
end

plot(firstsordered,secondsorted,'.')
xlabel('Region population')
ylabel('Price coupling parameters by region population')
title('Price coupling parameters as a function of population per region')