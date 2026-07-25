function q = row_quantile(values,p)
%ROW_QUANTILE Quantile of each row without toolbox dependencies.

validateattributes(p,{'numeric'},{'scalar','>=',0,'<=',1});

sorted_values = sort(values,2);
n = size(sorted_values,2);

if n == 0
    q = nan(size(values,1),1);
    return;
end

position = 1+(n-1)*p;
lower_idx = floor(position);
upper_idx = ceil(position);
weight = position-lower_idx;

if lower_idx == upper_idx
    q = sorted_values(:,lower_idx);
else
    q = (1-weight)*sorted_values(:,lower_idx)+weight*sorted_values(:,upper_idx);
end
end
