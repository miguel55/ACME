function firstNetwork=positionNeutrophil(dataIn,currFrameExternal)
%function firstNetwork=positionNeutrophil(dataIn,currFrameExternal)
%
%--------------------------------------------------------------------------
% positionNeutrophil  calculates the first metrics that will form handles.nodeNetwork:
%     first elements of the tracking process, get X, Y, Z for each neutrophil, then:
%       4      - label at frame
%       5      - frame
%       6      - unique label in 4D volume
%       7      - volume of neutrophil
%       8:13  - bounding box of neutrophil%
%       INPUT
%         dataIn:               path to labelled data
%
%         currFrameExternal:    currentFrame
%
%       OUTPUT
%         firstNetwork:         first network matrix as described above
%          
%--------------------------------------------------------------------------
% Code based on PhagoSight
%--------------------------------------------------------------------------%
%     Copyright (C) 2012  Constantino Carlos Reyes-Aldasoro
%
%     This file is part of the PhagoSight package.
%
%     The PhagoSight package is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, version 3 of the License.
%
%     The PhagoSight package is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with the PhagoSight package.  If not, see <http://www.gnu.org/licenses/>.
%
%--------------------------------------------------------------------------
%
% This m-file is part of the PhagoSight package used to analyse fluorescent phagocytes
% as observed through confocal or multiphoton microscopes.  For a comprehensive 
% user manual, please visit:
%
%           http://www.phagosight.org.uk
%
% Please feel welcome to use, adapt or modify the files. If you can improve
% the performance of any other algorithm please contact us so that we can
% update the package accordingly.
%
%--------------------------------------------------------------------------
%
% The authors shall not be liable for any errors or responsibility for the 
% accuracy, completeness, or usefulness of any information, or method in the content, or for any 
% actions taken in reliance thereon.
%
%--------------------------------------------------------------------------

firstNetwork=[];


if isa(dataIn,'char')
    % --------------------------- dataInName is not mat file, should be a folder with a)matlab files
    dir1                                    = dir(strcat(dataIn,'/*.mat'));
    numFrames                               = size(dir1,1);
    for counterDir=1:numFrames
        tempDir                             = dir1(counterDir).name;
        dataInName                          =  strcat(dataIn,'/',tempDir);
        
        dataFromFile                        =  load(dataInName);
        
        if isfield(dataFromFile,'dataIn')
            dataIn2                         = dataFromFile.dataIn;
        elseif isfield(dataFromFile,'dataL')
            dataIn2 = dataFromFile.dataL;
        else
            namesF                          = fieldnames(dataFromFile);
            dataIn2                         = dataFromFile.dataL;
        end

        currentCentroids3                   = positionNeutrophil(dataIn2,counterDir);
        firstNetwork                        = [firstNetwork;currentCentroids3];  %#ok<AGROW>
    end
    if (size(firstNetwork,1) == 0)
       firstNetwork = zeros(0,13); 
    else
        % Cell indexes
        firstNetwork(:,6)         = 1:size(firstNetwork,1);
    end
else
    numFrames=size(dataIn,4);
    for countFrames=1:numFrames
        %---- get properties of the labelled region
        currentProps                        =  regionprops(dataIn(:,:,:,countFrames),'Centroid','Area','BoundingBox');
        
        if ~isempty(currentProps)
            [rows,cols,levs]                = size(dataIn);
            %-----  arrange centroid and volume as a vertical vector
            currentCentroids                = [currentProps.Centroid]';
            currentVolume                 	= [currentProps.Area]';

            if levs>1
                currentBox                      = reshape([currentProps.BoundingBox],[6 size(currentProps,1)])';
                %-----  centroids in [Y X Z]
                currentCentroids3               = [currentCentroids(2:3:end) currentCentroids(1:3:end) currentCentroids(3:3:end)];
            else
                currentBox                      = reshape([currentProps.BoundingBox],[4 size(currentProps,1)])';
                %-----  centroids in [Y X Z]
                currentCentroids3               = [currentCentroids(2:2:end) currentCentroids(1:2:end) ones(size(currentProps,1),1)];
            end
            if (exist('currFrameExternal','var'))&&(~isempty(currFrameExternal))
               currentCentroids3(:,5)       =  currFrameExternal;           
            else
                currentCentroids3(:,5)      =  countFrames;
            end
            %----- keep unique label at the current frame
            currentCentroids3(:,4)         = (1:length(currentProps));

            %----- keep unique label at the current frame
            currentCentroids3(:,7)         =  currentVolume;
            
            if levs>1
                currentCentroids3(:,8:13)      = currentBox;
            else
                currentCentroids3(:,[8 9 11 12])      = currentBox;
                currentCentroids3(:,10)      = 1;
                currentCentroids3(:,13)      = 1;
            end
            %----- append if applicable
            firstNetwork                = [firstNetwork;currentCentroids3];  %#ok<AGROW>
        end
        clear currentProps currentCe* currentVo*;
    end

end
end


