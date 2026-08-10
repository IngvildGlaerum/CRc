function spikes_filtered = speed_filter(velocity,speed_threshold,position,spikes,frame_rate)
% velocity = similar length to position
% position = [time,x-pos,y-pos]
% speed_theshold = whatever threshold you decide: 0.05
% spikes = in spike times
% frame_rate = frame rate of the camera in Hz.
    % 2,5 cm/s cut-off (velocity is in m/s)
    not_moving = velocity < speed_threshold;

    reference = position(not_moving,1);

    % Calculate the absolute differences between each spike and all reference values
    differences = abs(spikes - reference);

    % Find the minimum difference for each spike
    min_differences = min(differences, [], 1);

    % Identify the spike times is simultaneous as non-movement
    spikes_to_keep = min_differences > frame_rate;

    % Keep only those spikes
    spikes_filtered = spikes(spikes_to_keep);

end
