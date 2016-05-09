function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
    %should be sqrt(400)= 20 in our example
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m, n] = size(X); %100 x 400 in our example
example_height = (n / example_width); %= 20 in example

% Compute number of items to display
display_rows = floor(sqrt(m)); %10 in ex.
display_cols = ceil(m / display_rows); %10 in ex.

% Between images padding
pad = 1;

% Setup blank display (for all 100 pictures)
% negative probably cause data in X are all negative
% pad on left or top + 10 * (20 + pad)
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows %10
	for i = 1:display_cols %10
		if curr_ex > m, %100
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch (10x10 patch for each image)
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
