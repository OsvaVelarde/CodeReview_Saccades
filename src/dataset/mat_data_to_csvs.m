%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in_folder = "/MATLAB Drive/metadata/";
out_folder = "/MATLAB Drive/output/";

files = dir(in_folder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% list_fields_rm = {'Audio', 'AudioSampleRate', ...
%                 'AudioNumBits', 'AudioNumChannels', ... 
%                 'AudioSamplesPerFrame', 'AudioSamplingRate', ... 
%                 'NaudioTime', 'NaudioFrames', 'NaudioSamples',...
%                 'videoFileName', 'Video',...
%                 'stimuli_id'};

csv_stim = {'FileName','VideoFormat',...
    'VideoFrameRate','VideoFrameTime',...
    'NvideoFrames',	'NrawVideoFrames','NumFrames_padded',...
    'NrawVideoTime','NvideoTime',...
    'stimFrameStart','stimFrameEnd',...	
    'stimDuration',...
    'stimStart','stimEnd',...		
    'blinkToStim_start_frames','blinkToStim_end_frames',...
    'blinkToStim_start_time','blinkToStim_end_time',...
    'blinkToblink_time','blinkToblink_frames'};

dict_stim = {'VideoSize','FrameStartTimes',...
            'FrameEndTimes','rawFrameStartTimes',...
            'rawFrameEndTimes','VideoPosition',...
            'VideoSize_original'};

csv_chapters = {'chapter_no',...
            'trigger_value_start','trigger_value_end',...
            'time_start','time_end',...
            'timestamp_difference_start','timestamp_difference_end',...
            'no_blinks','no_saccades','no_fixations'};

csv_saccades = {'pos_start_pixels_x', 'pos_start_pixels_y',...
                'pos_end_pixels_x', 'pos_end_pixels_y',...
                'pos_start_vda_x', 'pos_start_vda_y',...
                'pos_end_vda_x', 'pos_end_vda_y',...
                'timestamps_rel_stim'};

csv_eye_sacc = {'timestamps_edf_start','timestamps_edf_end',...
            'sampleidx_edf',...
            'duration_samples', 'duration_time',...
            'pos_start_x', 'pos_start_y',...
            'pos_end_x', 'pos_end_y',...
            'resolution_x', 'resolution_y'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ii = 3:length(files)
    filename = files(ii).name(1:end-4);
    load(strcat(in_folder,filename,'.mat'));

    if size(meta_stim.movie,1)==0
        continue
    end

    stim = meta_stim.movie{1,1};
    num_chapters = numel(stim);

    array_dicts = cell(1, num_chapters);

    % Stimuli - JSON file - dictionaries -----------------------   
    for jj = 1:num_chapters
        dict_element = struct();
        for ff = 1:numel(dict_stim)
            field = dict_stim{ff};
            dict_element.(field) = stim(ff).(field);
        end
        array_dicts{jj} = dict_element;
    end

    jsonString = jsonencode(array_dicts,"PrettyPrint",true);
    fid = fopen(strcat(out_folder,filename,'_stim.json'), 'w');
    fwrite(fid, jsonString, 'char');
    fclose(fid);

    % Stimuli - CVS file - tables ------------------------------
    csv_table = struct2table(stim);
    csv_table = csv_table(:, csv_stim);
    writetable(csv_table, strcat(out_folder,filename,'_stim.csv'));

    % Metadata Segments Movie ----------------------------------
    segms = metadata.segments.movie;
    segments_table = struct2table(segms);
    segments_table = segments_table(:, csv_chapters);
    writetable(segments_table, strcat(out_folder,filename,'_chapters.csv'));

    % Saccades -------------------------------------------------

    for jj = 1:num_chapters
        saccades_struct = segms(jj).saccades;
        saccades = struct();
        for ff = 1:numel(csv_saccades)
            field = csv_saccades{ff};
            saccades.(field) = saccades_struct.(field);
        end

        saccades_mtx = cell2mat(struct2cell(saccades)');

        if size(saccades_mtx,1) == 0
            continue
        end

        saccades_table = array2table(saccades_mtx, 'VariableNames', [csv_saccades, {'timestamps_rel_stim_end'}]);
        writetable(saccades_table, strcat(out_folder,filename,'_sacc_chap_',sprintf('%2d',jj),'.csv'));
    end

    % Eye.Saccades --------------------------------------------
    mtx_eye_saccade = cell2mat(struct2cell(metadata.eye.saccades)'); 
    table_eye_saccade = array2table(mtx_eye_saccade, 'VariableNames', csv_eye_sacc);   
    writetable(saccades_table, strcat(out_folder,filename,'_sacc.csv'));

end

	