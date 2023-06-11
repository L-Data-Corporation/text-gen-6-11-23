from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import time
from memory import custom_generate_chat_prompt


def encode(prompt, tokenizer, persona_length=0, max_new_tokens=0, add_special_tokens=True):

    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=add_special_tokens)

    # This is a hack for making replies more creative.
    if input_ids[0][0] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]

    # Llama adds this extra token when the first character is '\n', and this
    # compromises the stopping criteria, so we just remove it
    if input_ids[0][0] == 29871:
        input_ids = input_ids[:, 1:]

    # Persona handled differently than convo context - context is truncated, persona is not

    prompt_max_length = tokenizer.model_max_length - max_new_tokens - persona_length

    # Handling context truncation - truncate to max prompt length, then truncate again when first \n is hit
    if persona_length>0 and len(input_ids[0])>prompt_max_length:
        input_ids = input_ids[:, -prompt_max_length:]
        truncation_point = ((input_ids[0]==29871).nonzero()[0])[0].item()
        input_ids = input_ids[:, truncation_point+1:]

    return input_ids


def get_reply_from_output_ids(tokenizer, output_ids, input_ids, params):
    
    new_tokens = len(output_ids) - len(input_ids[0])
    reply = tokenizer.decode(output_ids[-new_tokens:])

    # Prevent LlamaTokenizer from skipping a space
    if len(output_ids) > 0:
        if tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
            reply = ' ' + reply

    return reply


class _SentinelTokenStoppingCriteria(StoppingCriteria):

    def __init__(self, sentinel_token_ids: list, starting_idx: int):
        StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx
        self.shortest = min([x.shape[-1] for x in sentinel_token_ids])

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            trimmed_len = trimmed_sample.shape[-1]
            if trimmed_len < self.shortest:
                continue

            for sentinel in self.sentinel_token_ids:
                sentinel_len = sentinel.shape[-1]
                if trimmed_len < sentinel_len:
                    continue

                window = trimmed_sample[-sentinel_len:]
                if torch.all(torch.eq(sentinel, window)):
                    return True

        return False


def generate_reply_HF(history, tokenizer, model, embeddings, params, stopping_strings=None):
    generate_params = {}
    for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'repetition_penalty', 'top_k']:
        generate_params[k] = params[k]

    persona = params['persona']
    # embeddings = params['embeddings']

    # might have to change this - make sure the history['You'][-1] part is accurate (list or dict etc)
    start_time = time.time()
    memory = custom_generate_chat_prompt(history['You'][-1], embeddings, history, params['character'],
        params['chunk_count', 'chunk_count_initial', 'time_weight'])
    print("Time to pull memory")
    print("--- %s seconds ---" % (time.time() - start_time))

    # add newline to persona ending if not already there
    if persona[-1]!= "\n": persona=persona+"\n"

    print("-------------------- Persona and memory --------------------")
    print(persona+memory)
    print("-------------------- History --------------------")
    print(history)

    # Encode the input - persona then context
    persona_and_mem_encoded = encode(persona+memory, tokenizer)
    history_encoded = encode(history, tokenizer, persona_length = len(persona_and_mem_encoded[0]), max_new_tokens = generate_params['max_new_tokens'])

    # Concat and send to cuda
    input_ids = torch.cat((persona_and_mem_encoded, history_encoded), 1)
    input_ids = input_ids.cuda()
    print("Prompt length: "+str(len(input_ids[0])))
    cuda = True

    # Add the encoded tokens to generate_params
    generate_params.update({'inputs': input_ids})

    # Find the eos tokens
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids

    # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
    stopping_criteria_list = StoppingCriteriaList()
    if type(stopping_strings) is list and len(stopping_strings) > 0:
        sentinel_token_ids = [encode(string, tokenizer).cuda() for string in stopping_strings]
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))

    # Update generate_params with stopping strings
    generate_params['stopping_criteria'] = stopping_criteria_list

    t0 = time.time()
    # Generate the entire reply at once.
    with torch.no_grad():
        output = model.generate(**generate_params)[0]
        if cuda:
            output = output.cuda()
    reply = get_reply_from_output_ids(tokenizer, output, input_ids, params)

    # Remove eos token
    if reply[-4:]=="</s>":
    	reply = reply[:-4]
    # Remove stopping criteria
    for stop_string in stopping_strings:
        if reply[-len(stop_string):]==stop_string:
            reply = reply[:-len(stop_string)]
    # Remove new line character at end
    if reply[-1]=="\n":
        reply = reply[:-1]

    t1 = time.time()

    new_tokens = len(output) - len(input_ids[0])
    print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens)')

    return reply