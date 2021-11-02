function [dataL,handles,statsData]=saveDataAndScores(handles,segm,scores)
%function [dataL,handles,statsData]=saveDataAndScores(handles,segm,scores)
%
%--------------------------------------------------------------------------
% saveDataAndScores generate the labelled images containing only cells
% and save them and the scores
%
%       INPUT
%         handles:      struct with all the parameters
%         segm:         segmentations
%         scores:       scores
%
%       OUTPUT
%         dataL:        data labelled from the threshold and small regions 
%                       removed
%         handles:      updated handles struct.
%         statsData:    statistics about labelled regions
%
%--------------------------------------------------------------------------
% Code based on PhagoSight
%--------------------------------------------------------------------------
%
%     Copyright (C) 2012  Constantino Carlos Reyes-Aldasoro
%
%     This file is part of the PhagoSight package.
%
%     The PhagoSight package is free software: you can redistribute it 
%     and/or modify it under the terms of the GNU General Public License 
%     as published by the Free Software Foundation, version 3 of the License.
%
%     The PhagoSight package is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with the PhagoSight package.  If not, see:
% 
%                   <http://www.gnu.org/licenses/>.
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
% accuracy, completeness, or usefulness of any information, or method in 
% the content, or for any actions taken in reliance thereon.
%
%--------------------------------------------------------------------------

numFrames = handles.numFrames;
mkdir(handles.dataLa)
if (nargin==3)
    handles.dataSc=[handles.dataLa(1:end-2) 'Sc'];
    mkdir(handles.dataSc)
end
for counterDir=1:numFrames
    dataOutName1 =  strcat(handles.dataLa,'/T',sprintf( '%05d',counterDir));
    dataL=segm(:,:,:,counterDir);
    if (nargin==3)
        dataOutName2 =  strcat(handles.dataSc,'/T',sprintf( '%05d',counterDir));
        dataSc=scores(:,:,:,counterDir);
    end
    if handles.minBlob~=0

        statsData = regionprops(dataL); 

        regS = regionprops(dataL,'area');

        [dataL,numNeutrop] = ...
            bwlabeln(ismember(...
                    dataL,...
                    find([regS.Area]>handles.minBlob)...
                    ));
        statsData = regionprops(dataL);

    else
        statsData = regionprops(dataL,'area'); 
    end

    save(dataOutName1,'dataL','numNeutrop','statsData');
    if (nargin==3)
        save(dataOutName2,'dataSc');
    end

end
dataL = handles.dataLa;
end
