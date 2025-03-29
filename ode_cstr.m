% This function is used by ODE45 in simulation of CSTR
function xdot = ode_cstr(t,x,flag,B,Da,beta,u)

x1 = x(1); x2 = x(2);
x1dot = [-x1 + Da*(1-x1)*exp(x2)];
x2dot = [-x2 + B*Da*(1-x1)*exp(x2)-beta*(x2-u)];

xdot = [x1dot;x2dot];