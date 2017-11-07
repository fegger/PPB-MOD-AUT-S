function [dydt]=sulfideKinetics(t,y,p)


k=p(1);
Ks=p(2);
Y=p(3);

S=y(1);
X=y(2);

dydt(1,1)=(-k.*S./(Ks+S).*X);
dydt(2,1)=(Y.*k.*S./(Ks+S).*X);
end

