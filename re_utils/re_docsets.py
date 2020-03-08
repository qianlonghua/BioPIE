from ner_utils.ner_docsets import *

# DocSets class for RE
class reDocSets(nerDocSets):
    # create relation types separately
    def create_relation_type_dicts(self):
        for fileset in self.filesets:
            fileset.create_relation_type_dict()
        return

    # only calculate statistics on entity mentions at sent- and docu- levels
    def calculate_docsets_relation_mention_statistics(self, levels=COUNT_LEVELS):
        ccounts = [[fileset.generate_docset_relation_mention_statistics(level) for level in levels] for fileset in self.filesets]
        return ccounts

    #
    def create_training_model(self, tcfg):
        if tcfg.bertID:
            if tcfg.bert_model is None:
                tcfg.bert_model = load_bert_model(tcfg.bert_path, tcfg.verbose)
            self.model = create_bert_classification_model('sent', tcfg)
        elif 'Lstm' in tcfg.model_name:
            self.model = build_lstm_model(level='sent', cfg=tcfg, EMBED_MATRIX=self.embed_matrix)
        else:
            self.model = build_cnn_model(level='sent', cfg=tcfg, EMBED_MATRIX=self.embed_matrix)

        if tcfg.verbose:  self.model.summary()
        return
