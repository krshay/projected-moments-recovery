function volume = nu_load_volume(volume_path,grid_size)
%NU_LOAD_VOLUME Load, crop, mask and resize a volume, matching the repo's
%   preprocessing so shell expansions are comparable with existing work.

[~,~,extension] = fileparts(volume_path);
switch lower(extension)
    case '.mat'
        contents = load(volume_path);
        names = fieldnames(contents);
        volume = contents.(names{1});
    otherwise
        volume = ReadMRC_local(volume_path);
end
volume = double(real(volume));
n = min(size(volume));
centre = floor((size(volume)-n)/2);
volume = volume(centre(1)+(1:n),centre(2)+(1:n),centre(3)+(1:n));
radius = (n-1)/2;
axis_points = (0:n-1)-radius;
[x,y,z] = ndgrid(axis_points,axis_points,axis_points);
volume(sqrt(x.^2+y.^2+z.^2) > radius) = 0;
if n ~= grid_size
    source = linspace(1,n,grid_size);
    [qx,qy,qz] = ndgrid(source,source,source);
    volume = interpn(1:n,1:n,1:n,volume,qx,qy,qz,'linear',0);
end
volume = volume/max(norm(volume(:)),eps);
end

function volume = ReadMRC_local(path)
fid = fopen(path,'r','ieee-le');
header = fread(fid,256,'int32=>int32');
nx = double(header(1)); ny = double(header(2)); nz = double(header(3));
mode = double(header(4));
fseek(fid,1024+double(header(24)),'bof');
switch mode
    case 0, data = fread(fid,nx*ny*nz,'int8=>double');
    case 1, data = fread(fid,nx*ny*nz,'int16=>double');
    case 2, data = fread(fid,nx*ny*nz,'single=>double');
    case 6, data = fread(fid,nx*ny*nz,'uint16=>double');
    otherwise, fclose(fid); error('Unsupported MRC mode %d',mode);
end
fclose(fid);
volume = reshape(data,nx,ny,nz);
end
