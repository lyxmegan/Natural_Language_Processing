def beam_search(self, sentence, beam_size=2):
       
        N = len(sentence)
        T = len(self.tag_indexer)
        beam_list = []
        pred_tags = []

        # Initialization step
        beam = Beam(beam_size)
        for y in range(T):
            tag = str(self.tag_indexer.get_object(y))
            if (isI(tag)):
                score = float("-inf")
            else:
                features = extract_emission_features(sentence,
                                                     0,
                                                     self.tag_indexer.get_object(y),
                                                     self.feature_indexer,
                                                     False)
                score = score_indexed_features(features, self.feature_weights)
            beam.add(self.tag_indexer.get_object(y), score)
        beam_list.append(beam)

        # Recursion step
        for x in range(1, N):
            beam = Beam(beam_size)
            for i in beam_list[x - 1].get_elts_and_scores():
                j = self.tag_indexer.index_of(i[0])
                for y in range(T):
                    # We want to ban out certain scenario:
                    # 1. We cannot have O, I tag sequence of any type
                    # 2. We cannot have I-x, I-y tag sequence of different types
                    # 3. We cannot have B-x, I-y tag sequence of any type of I other than x
                    prev_tag = str(j)
                    curr_tag = str(self.tag_indexer.get_object(y))
                    if (isO(prev_tag) and isI(curr_tag)) and \
                            (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) and \
                            (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
                        score = float("-inf")
                    else:
                        features = extract_emission_features(sentence,
                                                             x,
                                                             self.tag_indexer.get_object(y),
                                                             self.feature_indexer,
                                                             add_to_indexer=False)
                        score = score_indexed_features(features, self.feature_weights)
                    beam.add(self.tag_indexer.get_object(y), i[1] + score)
            beam_list.append(beam)

        # Backtrace
        beam_list = reversed(beam_list)
        for beam in beam_list:
            pred_tags.append(beam.head())

        pred_tags = list(reversed(pred_tags))

        return LabeledSentence(sentence, chunks_from_bio_tag_seq(pred_tags))
