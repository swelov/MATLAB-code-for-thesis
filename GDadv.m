clear all
close all

% A matrix of the house prices, defined by definition transaction price
prices = dlmread('Dataminus2bornholmlasoarofano.txt'); %First column is an index column
%Running shows that  at row 3 (Frederiksberg) column 65 (2007Q4) data is missing; I add the
%aritmetic mean between time-adjacent prices: 30994 + 39382 = 35188
price = prices(:,2:(length(prices)-16)); %The price matrix. Each columns is the prices for a given time.
priceall = prices(:,2:length(prices));

%----------Diagram with 2 y-axes-------------
%M1all = dlmread('M1.txt');
%M1al = M1all(:,1);
%M1 = M1al(1 : 3 : end);
%x = 1992:0.25:2018.75;
%grid
%hold on

%yyaxis left
%y3 = mean(priceall);
%plot(x,y3,'b-.');
%ylim([0 1.06*max(y3)])
%xlim([1992 2019])
%ylabel('Average house price (dkr/m^2)')
%xlabel('Year')


%yyaxis right
%y2 = M1;
%plot(x,M1);
%ylabel('Money supply M1 (dkr)')
%ylim([0 1.06*max(M1)])
%legend('Average house price')
%title('The average house price and money supply in Denmark')
%hold off
%----------END Diagram with 2 y-axes---------


%priceadd1 = dlmread('Other93GDBS.txt');
%price(:,93) = priceadd1;
%priceadd2 = dlmread('Other94GDBS.txt');
%price(:,94) = 10000*priceadd2;
%priceadd3 = dlmread('Other95GDBS.txt');
%price(:,95) = 10000*priceadd3;
%priceadd4 = dlmread('Other96GDBS.txt');
%price(:,96) = 10000*priceadd4;
%priceadd5 = dlmread('Other97GDBS.txt');
%price(:,97) = 10000*priceadd5;
%priceadd6 = dlmread('Other98GDBS.txt');
%price(:,98) = 10000*priceadd6;
%priceadd7 = dlmread('Other99GDBS.txt');
%price(:,99) = 10000*priceadd7;
%priceadd8 = dlmread('Other100GDBS.txt');
%price(:,100) = 10000*priceadd8;
%priceadd9 = dlmread('Other101GDBS.txt');
%price(:,101) = 10000*priceadd9;
%priceadd10 = dlmread('Other102GDBS.txt');
%price(:,102) = 10000*priceadd10;
%priceadd11 = dlmread('Other103GDBS.txt');
%price(:,103) = 10000*priceadd11;
%priceadd12 = dlmread('Other104GDBS.txt');
%price(:,104) = 10000*priceadd12;
%priceadd13 = dlmread('Other105GDBS.txt');
%price(:,105) = 10000*priceadd13;
%priceadd14 = dlmread('Other106GDBS.txt');
%price(:,106) = 10000*priceadd14;
%priceadd15 = dlmread('Other107GDBS.txt');
%price(:,107) = 10000*priceadd15;
%priceadd16 = dlmread('Other108GDBS.txt');
%price(:,108) = 10000*priceadd16;
%--------------------------------------------
%priceadd1 = dlmread('Pred93GDBS.txt');
%price(:,93) = 10000*priceadd1;
%priceadd2 = dlmread('Pred94GDBS.txt');
%price(:,94) = 10000*priceadd2;
%priceadd3 = dlmread('Pred95GDBS.txt');
%price(:,95) = 10000*priceadd3;
%priceadd4 = dlmread('Pred96GDBS.txt');
%price(:,96) = 10000*priceadd4;
%priceadd5 = dlmread('Pred97GDBS.txt');
%price(:,97) = 10000*priceadd5;
%priceadd6 = dlmread('Pred98GDBS.txt');
%price(:,98) = 10000*priceadd6;
%priceadd7 = dlmread('Pred99GDBS.txt');
%price(:,99) = 10000*priceadd7;
%priceadd8 = dlmread('Pred100GDBS.txt');
%price(:,100) = 10000*priceadd8;
%priceadd9 = dlmread('Pred101GDBS.txt');
%price(:,101) = 10000*priceadd9;
%priceadd10 = dlmread('Pred102GDBS.txt');
%price(:,102) = 10000*priceadd10;
%priceadd11 = dlmread('Pred103GDBS.txt');
%price(:,103) = 10000*priceadd11;
%priceadd12 = dlmread('Pred104GDBS.txt');
%price(:,104) = 10000*priceadd12;
%priceadd13 = dlmread('Pred105GDBS.txt');
%price(:,105) = 10000*priceadd13;
%priceadd14 = dlmread('Pred106GDBS.txt');
%price(:,106) = 10000*priceadd14;
%priceadd15 = dlmread('Pred107GDBS.txt');
%price(:,107) = 10000*priceadd15;
%priceadd16 = dlmread('Pred108GDBS.txt');
%price(:,108) = 10000*priceadd16;

%Calculating the average inflation (and average price) in order to get Beta
avprice = mean(price); %Mean price of the regions of every time
avcalc = avprice(2:length(avprice)); %All average prices except the first price at first time unit
inflation = avcalc./avprice(1:length(avprice)-1); %Average inflation 
inf = mean(inflation)-1; %Average inflation per time unit (3 months)
avpricetot = mean(avprice); %Average price for all regions and all times
infprice = inf*avpricetot; %This is the average price increase per unit time

%plot(1:106,prices(88,2:107)) 
%Check how the prices eveloves with time

%The equations to solve are: x(i,t+1) = a(i) + b*t + sum[c(i,j)* x(i,t))
%for all regions j, in every region i.
r = length(price(:,1)); %Number of regions 
t = 1:size(price,2); %Vector with number of time intervalls of 3 months
nt= size(price,2); %Number of time intervals
%a = 2700*ones(r,1); %Preliminary Alpha constants vector a(i)
a = 0*0.5*price(:,1);
%b = 0*0.1*infprice*ones(r,1); %Preliminary Beta constant vector (b) everywhere
%b = 44*ones(r,1);
b = 0*0.5*(price(:,92)./price(:,1)).^(1/91);
c = 0*0.5*(0.5/92 * ones(r) + eye(r)*0.5); %Preliminary matrix of all coupling constants c(i,j)
%c(i,j) represents the price coupling on region i due to region j

%---------------------CALCULATE PRICES--------------------
%Let x represent the matrix of propagated prices one time step; x(r,t) is
%the propagated price at region r and time t (the data is taken from the
%prices at t-1 acostfnd at the first time we keep 0's). First we define its size with 0's:
x = zeros(r,nt);
%x(:,2) = a + b * t(1) + c*x(:,1);
x(:,1)=price(:,1);
for k = 1:18000 %Number of iterations that we change
%Now we will propagate all prices (for all regions and all times except last time) one time unit 
%according to the equation, and calculate the errors. We calculate from the
%2nd time step as we need the data from the first time:
for i=2:nt %2:length(t)
    x(:,i) = a + b*t(i-1) + c*price(:,i-1);
    %costf(:,i-1) = (x(:,i)-price(:,i)).^2; %The cost function matrix (region by region) 
    %shows the squared errors
end
error = x-price;
costf = error.^2; %The sum of these elements is the cost function that is gonna be minimized
%sum(costf);
%accuracy =(costf.^(1/2))./price;
%costf = (x(:,2)-price(:,2)).^2; %The cost function vector (region by region)
%Is the average difference (as part of price) 
%between the propagated price 1 step in time and the price

%mean(mean(accuracy)) 

%---------------------CALCULATE DERIVATIVES----------------------------
%Now let us calculate the partial derivatives of the cost function with respect to all parameters 

%Let's start with the partial derivatives of costf with respect to alpha: all
%a(i).
%Since costf(i,:) only depends on a(i) and vanishes for a(j), j different from i, the
% derivative of each element in each line costf(i,:) with respect to all
% elements of a, will vanish except for a(i). 

dcostfa = sum(2*error(:,2:(nt)),2); %We skip the first column, it 
%does not represent any propagated price.
% The sum of each row i of the errors is proportional to the derivative of and how much change
%should be done to the corresponding row element of a (a(i)) :
%a -> a - p*dcostfa, p small number

%Now to the derivatives of costf(r,t) with repsect to beta, b: it's
%2*t*error(r,t): For each time propagation of all prices at different
%times, we only need the average of all 2*error(c1,c2)*t(c2), but not from the first column:

dcostfb = 2*error(:,2:nt)*(t(2:nt)'); 

%Now to the derivative of costf with respect to c(c1,c2), the coupling
%matrix.
%How much does c(c1,c2) change the cost function?

%The matricies we work with, to establish the derivative of the cost function with respect to 
%the coupling matrix c,(dcostfc), does not use first time of error (not propagated), and not 
%last time of x:

errorcalc = error(:,2:nt);
xcalc = x(:,1:(nt-1));

%Now dcostfc defined as the matrix of summed elements of the derivatives of the cost function with
%respect to all coupling constants.
dcostfc = errorcalc*transpose(xcalc);
%---------------------END OF DERIVATIVE CALCULATION-----------------------

costftot(k) = sum(costf(:,2:nt), 'all');

%Find how many parameters are positive and negative
indicesa = find(a<0);
aNeg = length(indicesa);
%aNeg = length(a)-aPos;
atest(k) = aNeg/93;

indicesb = find(b<0);
bNeg = length(indicesb);
%bNeg = length(b)-bPos;
btest(k) = bNeg/93;

s=sign(c);
ipositif=sum(s(:)==1);
inegatif=sum(s(:)==-1);
ctest(k) = inegatif/(93*93);

%Diagonal elements of c
indiciesdiagc = find(diag(c)<0);
cNeg = length(indiciesdiagc);
%cNeg = length(diag(c))-cPos;
cdiagtest(k) = cNeg/93;

% Size of diagonal elements compared to all elements
diagcacc(k) = mean(mean(c))/mean(diag(c));

% Save parameters
ar(:,k) = a;
br(:,k) = b;
cr(:,:,k) = c;
cav(k) = mean(mean(c));
cdav(k) = mean(diag(c));
%Let's introduce the learning rate: L
L = 0.00000001/(r*(nt-1));
%The new constants are now simply
a = a - 15000000*L*dcostfa; %15
b = b - 700000*L*dcostfb;   %7
c = c - 0.5*L*dcostfc;  %0.5
%a = a - 0.5*L*dcostfa;
%b = b - 0.5*L*dcostfb;
%c = c - 0.5*L*dcostfc;

end
%Getting the MSD from sum of costf:
mse = costftot/(r*(nt-1));
rmse = mse.^0.5; 
fel=rmse/avpricetot;%Root-mean-square-error relative to the average price
%plot(1:k,atot,1:k,btot,'g--',1:k,ctot,'r:')
fel(k)
%imagesc(c)
 %colorbar
% Supervising results
test(1,:) = fel;
test(2,:) = atest;
test(3,:) = btest;
test(4,:) = ctest;
test(5,:) = cdiagtest;
%test(6,:) = diagcacc;


%plot(1:(length(costftot)),test(1,:),'',1:(length(costftot)), test(2,:),'--g',1:(length(costftot)),test(3,:),'r',1:(length(costftot)),test(4,:),'-.k',1:(length(costftot)),test(5,:),'c') %Does cost decrease?
%xlabel('Number of gradient descent iterations') 
%ylabel('RMSE per Average price and fractions of negative parameters')
%legend('Root-mean-square-error per average price','Fraction of negative \alpha-parameters','Fraction of negative \beta-parameters','Fraction of negative \gamma-parameters','Fraction of negative diagonal-\gamma-parameters')

%plot(1:(length(costftot)), mean(ar),'--g',1:(length(costftot)),mean(br),'r',1:(length(costftot)),cav,'-.k',1:(length(costftot)),cdav,'c') %Does cost decrease?
%xlabel('Number of gradient descent iterations') 
%ylabel('The average of the \alpha, \beta and \gamma-parameters')
%legend('Average \alpha-parameter value','Average \beta-parameter value','Average \gamma-parameter value','Average diagonal-\gamma-parameters')
%title('Average parameter values as a function of number of iterations')

MeanA = mean(ar(:,k));
MeanB = mean(br(:,k));
MeanC = mean(mean(cr(:,:,k)));
MeanCdiag = mean(diag(cr(:,:,k)));
results(:,1) = test(:,k);
results(1,2) = test(1,k);
results(2,2) = MeanA;
results(3,2) = MeanB;%Fixa en till rad i test för att se genomsnittet av parametrarna också
results(4,2) = MeanC;
results(5,2) = MeanCdiag;
results;
%----IMAGE and DISTRIBUTION of C-matrix---
%imagesc(c)
%colorbar
%title('Price coupling parameter value visualized by color');
%xlabel('Region index')
%ylabel('Region index')
%legend('Distribution of price coupling parameter values')

%c-fördelning
%histfit(reshape(c,[],1))
%xlabel('Parameter value')
%ylabel('Number of parameters')
%title('Distribution of price coupling parameter values')
%----END IMAGE and DISTRIBUTION of C-matrix---

%STD + Mean of c:
std(reshape(c,[],1));
%STD of non-diagonal c:
nondiagc = c-diag(diag(c));
stdnondiag = std(reshape(nondiagc,[],1));
StdNonD = (stdnondiag^2)*(r^2)/(r*(r-1));
MeanNonD = mean(mean(nondiagc))*(r^2)/(r*(r-1));
%plot(1:length(costftot),fel(1:length(costftot)))
%ylabel('Average error after one time propagation of prices')
%xlabel('Number of gradient descent iterations') 

%---------------CHECK distance correlation-------------
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
mesh(DM)
colorbar
xlabel('Region index')
ylabel('Region index')
zlabel('Distance (km)')
title('Distance between regions represented in km and by colour')

%Check correlation and more of distance vs coupling constant
cvector = reshape(c,[],1);
%abscvector= abs(cvector);
DMvector = reshape(DM,[],1);
[firstsorodered, firstsortorder] = sort(DMvector);
secondsorted = cvector(firstsortorder);

plot(firstsorodered,secondsorted,'.')
xlabel('Distance between pair of regions (km)')
ylabel('Price coupling parameters')
title('Price coupling parameters versus the distance between the regions')

%corrcoef(firstsortordered,secondsorted);
%---------------END CHECK distance correlation-------------

%------Check population correlation------------
Pvector = 1000/4*(populations(:,2) + populations(:,4) + populations(:,6) + populations(:,8))+ 1/4*(populations(:,3) + populations(:,5) + populations(:,7) + populations(:,9));
[firstordered, firstsortorder] = sort(Pvector);
secondsorted = cvector(firstsortorder);

for i = 1:r
    c1 = c(i,:);
    secondsorted = c1(firstsortorder);
    info = corrcoef(firstordered,secondsorted);
    corr(i) = info(2,1);
end

%plot(firstordered,secondsorted)
%xlabel('Region population')
%ylabel('Price coupling parameters by region population')
%legend('Price coupling parameters as a function of population per region')



plot(firstordered,secondsorted,'.')
xlabel('Region population')
ylabel('Price coupling parameters by region population')
title('Price coupling parameters versus population per region')
%-----END Check population correlation------------



%plot(1:length(costftot),cdertid) %Does derivative approach 0?
%plot(1:length(btid),btid)
%ylabel('Beta')
%xlabel('Number of gradient descent iterations') 
%legend('L = 0.000000007/(r*(nt-1)), a = 200*ones(r,1)
%b = 0.0699*infprice*ones(r,1); 
%c = 0.01 * ones(r) + eye(r)*0.5; 


%figure;imagesc(abs(error(:,2:nt)))

%plot(1:(nt-1),min(abs(error(:,2:nt))))
%ylabel('Minimal error ')


%----------PREDICTIONS----------
%Adding the Bootstrapped predicted valuesto price:
%priceadd1 = dlmread('Pred93GDBS.txt');
%price(:,93) = 10000*priceadd1;
%priceadd2 = dlmread('Pred94GDBS.txt');
%price(:,94) = 10000*priceadd2;
%priceadd3 = dlmread('Pred95GDBS.txt');
%price(:,95) = 10000*priceadd3;
%priceadd4 = dlmread('Pred96GDBS.txt');
%price(:,96) = 10000*priceadd4;
%priceadd5 = dlmread('Pred97GDBS.txt');
%price(:,97) = 10000*priceadd5;
%priceadd6 = dlmread('Pred98GDBS.txt');
%price(:,98) = 10000*priceadd6;
%priceadd7 = dlmread('Pred99GDBS.txt');
%price(:,99) = 10000*priceadd7;
%priceadd8 = dlmread('Pred100GDBS.txt');
%price(:,100) = 10000*priceadd8;
%priceadd9 = dlmread('Pred101GDBS.txt');
%price(:,101) = 10000*priceadd9;
%priceadd10 = dlmread('Pred102GDBS.txt');
%price(:,102) = 10000*priceadd10;
%priceadd11 = dlmread('Pred103GDBS.txt');
%price(:,103) = 10000*priceadd11;
%priceadd12 = dlmread('Pred104GDBS.txt');
%price(:,104) = 10000*priceadd12;
%priceadd13 = dlmread('Pred105GDBS.txt');
%price(:,105) = 10000*priceadd13;
%priceadd14 = dlmread('Pred106GDBS.txt');
%price(:,106) = 10000*priceadd14;
%priceadd15 = dlmread('Pred107GDBS.txt');
%price(:,107) = 10000*priceadd15;
%priceadd16 = dlmread('Pred108GDBS.txt');
%price(:,108) = 10000*priceadd16;
t = 1:108;
s = 93; %Starting value
f = 108; % Finishing value
x(:,s-1) = price(:,s-1);
for i=s:f
    x(:,i) = a + b*t(i-1) + c*x(:,i-1);
   error(:,i) = x(:,i)-priceall(:,i);
    costf(:,i) = error(:,i).^2; %The cost function matrix (region by region) 
    %shows the squared errors
end

%----------END PREDICTIONS----------

%----------STATISTICS PREDICTIONS----------
s = 77; f=77;
error(:,s:f) = price(:,s:f)-priceall(:,s:f);
costf(:,s:f) = error(:,s:f).^2; %The sum of these elements is the cost function that is gonna be minimized

MSE = mean(mean(costf(:,s:f)));
MAE  = mean(mean(abs(error(:,s:f))));
MAPE = 100*mean(mean(abs(error(:,s:f))./priceall(:,s:f)));
ME   = mean(mean(error(:,s:f)));
MPE  = 100*mean(mean(error(:,s:f)./priceall(:,s:f)));

stat(1) = MSE;
stat(2) = MAE;
stat(3) = MAPE;
stat(4) = ME;
stat(5) = MPE;

%----------END STATISTICS PREDICTIONS----------

%----------PROFIT STATISTICS--------------
%1. How much does the prediction say we will earn
p1 = (price(:,f)-priceall(:,s-1))./priceall(:,s-1); %Vector with predicted %-change in price for every region after the period
Positive = p1 > 0.0;
p1(~Positive) = 0; %Vector with only positive predicted profits, the negative predictions are 0.
p1 = nonzeros(p1);
sort(p1)
preprofit = mean(p1) %Predicted profit by investing equally much in all regions that predicts increase in price


r1 = (priceall(:,f)-priceall(:,s-1))./priceall(:,s-1); %Real price profit in all regions during the same time
r1(~Positive)=0; %Vector with the real predicted profits
r1 = nonzeros(r1);
realprofit = mean(r1)
%----------END PROFIT STATISTICS-------------

%--------PLOT PREDICTION ERRORS------------
mesh(error(:,93:108))
xlabel('Time/year')
ylabel('Region index')
xticklabels({'2015','2016','2017','2018','2019'})
zlabel('House price (dkk/m^2)')
title('Errors of predictions in validation period')
colorbar

%errorv = reshape(error(:,93:108),[],1);
%histfit(errorv)
%xlabel('Error/dkk/m^2') 
%ylabel('Number of predictions')
%title('Distribution of errors for predictions in validation period')