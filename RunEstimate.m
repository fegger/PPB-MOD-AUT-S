%%
clear
clc
close all
%%

filename = 'H:\02_Projects\02_PPB-GRO-BAT\04_Experiments\06_PPB-GAS-GRO-BAT-Run4-24-10-2017\02_Data\PPB-GAS-GRO-BAT-Run4-DailyAnalysis-24-10-2017.csv';
delimiter = ',';
startRow = 6;

formatSpec = '%*s%*s%f%*s%*s%f%*s%f%f%f%f%*s%f%f%*s%*s%*s%f%f%f%f%f%*s%[^\n\r]';

fileID = fopen(filename,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');

fclose(fileID);

data = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
%%
time=data(5:4:end,1);
t1=data(5:4:end,3);
t2=data(6:4:end,3);
t3=data(7:4:end,3);
td=data(8:4:end,3);
scod1=data(5:4:end,13);
scod2=data(6:4:end,13);
scod3=data(7:4:end,13);
tcod1=data(5:4:end,12);
tcod2=data(6:4:end,12);
tcod3=data(7:4:end,12);
p1=data(5:4:end,12)-data(5:4:end,13);
p2=data(6:4:end,12)-data(6:4:end,13);
p3=data(7:4:end,12)-data(7:4:end,13);
pd=data(8:4:end,12)-data(8:4:end,13);
T=[t1.*2 t2.*2 t3.*2];
scod=[scod1 scod2 scod3];
tcod=[tcod1 tcod2 tcod3];
P=[p1 p2 p3];
Tm=mean(T,2);
Pm=mean(P,2);
sSulf=std(T,0,2);
sSCOD=std(scod,0,2);
sTCOD=std(tcod,0,2);
covar=zeros(length(scod),1);
for i=1:length(scod)
    covarMatrix=cov(tcod(i,:),scod(i,:));
    covar(i,1)=covarMatrix(1,2);
end
sPCOD=sqrt(sTCOD.^2+sSCOD.^2-2.*covar);
e95t=tinv(0.975,2)/sqrt(3).*sSulf;
e95p=tinv(0.975,2)/sqrt(3).*sPCOD;

tdcod=td.*2;

%%
k0=0.1; %0.02 pm 0.008
Ks0=50; %40 pm 20
Y0=0.8; %0.8 pm 0.04
p0(1)=k0;
p0(2)=Ks0;
p0(3)=Y0;

yIni=[Tm(1,1);Pm(1,1)];
%yIni=[200;220];
%[x,resnorm,resid,exitflag,output,lambda,J] = lsqcurvefit(@optimSulfide,p0,time,[Tm,Pm]);
options=optimoptions(@lsqnonlin,'FunctionTolerance',1e-6);
[x,resnorm,resid,exitflag,output,lambda,J] = lsqnonlin(@optimSulfide,p0,[],[],options,time,[Tm,Pm],[sSulf,sPCOD],yIni);
ci = nlparci(x,resid,'jacobian',J);

%%

hold on
errorbar(time,Tm,e95t,'-d');
%errorbar(time,Cm,e95c,'-x');
errorbar(time,Pm,e95p,'-*');
plot(time,tdcod,'--o');
%plot(time,td,'--v');
plot(time,pd,'-->');


