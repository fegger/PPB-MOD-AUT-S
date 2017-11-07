function f=optimSulfide(p,tspan,ySys,s,yIni)

options = odeset('RelTol',1e-4,'Stats','on','OutputFcn',@odeplot);
[t,y,p]=ode15s(@sulfideKinetics,tspan,yIni,options,p);

f=sqrt(abs(1./s)).*(y(:,1:2)-ySys);

