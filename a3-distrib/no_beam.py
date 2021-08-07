def decode(self, sentence):
        pred_tags = []
        N = len(sentence)
        T = len(self.tag_indexer)
        v = np.zeros(shape=(T, N))
        max_prev = np.zeros(shape=(T, N))

        score_matrix = np.zeros(shape=(T, N))
        for y in range(T):
            for i in range(N):
                features = extract_emission_features(sentence,
                                                     i,
                                                     self.tag_indexer.get_object(y),
                                                     self.feature_indexer,
                                                     add_to_indexer=False)
                score = sum([self.feature_weights[i] for i in features])
                score_matrix[y, i] = score

        # Initialization step
        for y in range(T):
            # "+" because the probabilities are log-based
            tag = str(self.tag_indexer.get_object(y))
            if (isI(tag)):
                v[y, 0] = float("-inf")
            else:
                v[y, 0] = score_matrix[y, 0]
            max_prev[y, 0] = 0

        # Recursion step
        for i in range(1, N):
            for y in range(T):
                #tmp1 = np.zeros(T)
                prev_prob = np.zeros(T)
                for y_prev in range(T):
                    # "+" because the probabilities are log-based
                    # We want to ban out certain scenario:
                    # 1. We cannot have O, I tag sequence of any type
                    # 2. We cannot have I-x, I-y tag sequence of different types
                    # 3. We cannot have B-x, I-y tag sequence of any type of I other than x
                    prev_tag = str(self.tag_indexer.get_object(y_prev))
                    curr_tag = str(self.tag_indexer.get_object(y))
                    if (isO(prev_tag) and isI(curr_tag)) or \
                            (isI(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)) or \
                            (isB(prev_tag) and isI(curr_tag) and get_tag_label(prev_tag) != get_tag_label(curr_tag)):
                        #tmp1[y_prev] = float("-inf")
                        prev_prob[y_prev] = float("-inf")
                    else:
                        #tmp1[y_prev] = v[y_prev, i - 1] + score_matrix[y, i]
                        prev_prob[y_prev] = v[y_prev, i - 1]
                v[y, i] = np.max(prev_prob) + score_matrix[y, i]
                max_prev[y, i] = np.argmax(prev_prob)
                # Termination step (skipped because we don't have the end state)
        # Backtrace
        pred_tags.append(self.tag_indexer.get_object(np.argmax(v[:, N - 1])))
        for i in range(1, N):
            pred_tags.append(self.tag_indexer.get_object(max_prev[self.tag_indexer.index_of(pred_tags[-1]), N - i]))

        pred_tags = list(reversed(pred_tags))

        

        return LabeledSentence(sentence, chunks_from_bio_tag_seq(pred_tags))