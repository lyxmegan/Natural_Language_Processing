class Seq2SeqSemanticParser(object):
    def __init__(self, model_input_emb, model_enc, model_output_emb, model_dec, args, output_indexer):
        self.model_input_emb = model_input_emb
        self.model_enc = model_enc
        self.model_output_emb = model_output_emb
        self.model_dec = model_dec
        self.args = args
        self.output_indexer = output_indexer

    def decode(self, test_data):
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = 1
        EOS_token = self.output_indexer.index_of('<EOS>')
        SOS_label = self.output_indexer.get_object(SOS_token)
        beam_length = 1
        derivations = []
        print("EOS_token: ", EOS_token)

        for ex in test_data:
            count = 0
            y_toks =[]
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([ex.x_len()])

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            dec_input = torch.as_tensor([[SOS_token]])

            while (dec_input.item() != EOS_token) and count <= self.args.decoder_len_limit:
                dec_output, dec_input, dec_input_val, dec_hidden = decode(dec_input, dec_hidden, self.model_output_emb, self.model_dec, beam_length)
                y_label = self.output_indexer.get_object(dec_input.item())
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                count = count + 1
                #print("dec_input: ", dec_input)
                #print("dec_input.item(): ", dec_input.item())
            derivations.append([Derivation(ex, 1.0 , y_toks)])
            #print("prediction: ", y_toks)
        return derivations