function recording_ids = get_recording_ids(protocol, client_id, phrase_id, rec_type, channel_type, pb_device)


num_of_rows = height(protocol);

matching_client = ones(num_of_rows, 1);
if (~isempty(client_id))
    matching_client = strcmp(protocol(:, :).client_id, client_id);
end

matching_phrase = ones(num_of_rows, 1);
if (~isempty(phrase_id))
    matching_phrase = strcmp(protocol(:, :).phrase_id, phrase_id);
end

switch rec_type
    case 'stored'
        matching_rec_type = strcmp(protocol(:, :).with_ir, 'True');
    case 'authentic'
        matching_rec_type = strcmp(protocol(:, :).with_ir, 'False');
    case 'playback'
        matching_rec_type = strcmp(protocol(:, :).rec_type, 'playback');
    otherwise
        matching_rec_type = ones(num_of_rows, 1);
end


matching_channel_type = ones(num_of_rows, 1);
if ~isempty(channel_type)
    matching_channel_type = strcmp(protocol(:, :).channel_type, channel_type);
end

matching_pb_device = ones(num_of_rows, 1);
if ~isempty(pb_device)
    matching_pb_device = strcmp(protocol(:, :).pb_device, pb_device);
end

recording_ids = protocol(matching_client & matching_phrase & matching_rec_type & matching_channel_type & matching_pb_device, :).rec_id;
