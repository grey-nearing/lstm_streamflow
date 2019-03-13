function [Ixy,Hx,Hy] = mutual_info(X,Y,Nbins)

X = X(:);
Y = Y(:);

% binning strategy
By = (min(Y)-1e-6):Bw:(max(Y)+1e-6+Bw);
Bx = (min(X)-1e-6):Bw:(max(X)+1e-6+Bw);

% remove missing values
Im = find(any(isnan([X,Y]),2));
X(Im) = [];
Y(Im) = [];

N = length(X);
assert(N==length(Y))

Mx = length(Bx);
My = length(By);

Pxy = zeros(Mx,My);
for n = 1:N
  x = X(n);
  y = Y(n);
  y = find(y<=By,1,'first');
 if isempty(y); keyboard; end
  x = find(x<=Bx,1,'first');
 if isempty(x); keyboard; end
  Pxy(x,y) = Pxy(x,y) + 1;
end
Pxy = Pxy/N;

%PPxy = zeros(Mx,My);
%for y = 2:My
% I = find(Y<=By(y)&Y>By(y-1));
% for x = 2:Mx
%  PPxy(x,y) = length(X(X(I)<=Bx(x)&X(I)>Bx(x-1)));
% end
%end
%PPxy = PPxy/N;
%max(abs(Pxy(:)-PPxy(:)))

Px  = squeeze(sum(Pxy,2));
Py  = squeeze(sum(Pxy,1));

Px = Px(:);
Py = Py(:);
Pxy = Pxy(:);

if abs(sum(Px)-1)>1/N^2;  error('Px does not sum to 1');  end;
if abs(sum(Py)-1)>1/N^2;  error('Py does not sum to 1');  end;
if abs(sum(Pxy)-1)>1/N^2; error('Pxy does not sum to 1'); end;

Hx = -Px(Px>0)'*log(Px(Px>0));
Hy = -Py(Py>0)'*log(Py(Py>0));

Hxy = -Pxy(Pxy>0)'*log(Pxy(Pxy>0));

Ixy = Hx+Hy-Hxy;






