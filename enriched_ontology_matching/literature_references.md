# Literature References

This note collects the core references that support the metric families used in the ontology-matching pipeline, and then separates out what is standard in the literature from what is a repo-specific derived score.

## Core References

1. Wu, Z., and Palmer, M. (1994). "Verb Semantics and Lexical Selection." ACL 1994.  
   Link: https://aclanthology.org/P94-1019/

   Why it matters here: this is the canonical Wu-Palmer reference for taxonomic similarity in WordNet. The repo's `wup` score is built from WordNet-style synset similarity, but the exact implementation in this repo is an engineered blend (`max_wup + avg_wup`) rather than the original paper's standalone formula.

2. Reimers, N., and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP-IJCNLP 2019.  
   Link: https://aclanthology.org/D19-1410/

   Why it matters here: this is the main reference for sentence embeddings that can be compared with cosine similarity. It supports the repo's `cosine_avg` metric, which uses Sentence-Transformers embeddings and cosine similarity.

3. Sentence-Transformers cross-encoder NLI model card for `cross-encoder/nli-MiniLM2-L6-H768`.  
   Link: https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768

   Why it matters here: this documents the NLI model used in the repo. The model is trained on SNLI and MultiNLI and returns contradiction, entailment, and neutral scores for a sentence pair. That directly supports the repo's directional entailment probabilities.

4. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT 2019.  
   Link: https://aclanthology.org/N19-1423/
   Semantic Scholar: https://www.semanticscholar.org/search?q=BERT%3A%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding&sort=relevance

   Why it matters here: BERT is the primary academic reference for the paired-input transformer classifier pattern used by many NLI cross-encoders. For MultiNLI, the model receives the premise and hypothesis together as one packed sequence and predicts the NLI label with a classifier head. That is the same cross-encoder style used by the repo's NLI layer, where both entity signatures are scored jointly rather than embedded independently.

5. Williams, A., Nangia, N., and Bowman, S. R. (2018). "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference." NAACL-HLT 2018.  
   Link: https://aclanthology.org/N18-1101/
   Semantic Scholar: https://www.semanticscholar.org/search?q=A%20Broad-Coverage%20Challenge%20Corpus%20for%20Sentence%20Understanding%20through%20Inference&sort=relevance

   Why it matters here: this is the Multi-Genre Natural Language Inference (MultiNLI/MNLI) reference. It defines a large premise-hypothesis entailment benchmark across multiple genres, which is directly relevant because the repo's cross-encoder model card reports training on MultiNLI.

6. Bowman, S. R., Angeli, G., Potts, C., and Manning, C. D. (2015). "A large annotated corpus for learning natural language inference." EMNLP 2015.  
   Link: https://aclanthology.org/D15-1075/
   Semantic Scholar: https://www.semanticscholar.org/search?q=A%20large%20annotated%20corpus%20for%20learning%20natural%20language%20inference&sort=relevance

   Why it matters here: this is the Stanford Natural Language Inference (SNLI) reference. SNLI established the large-scale labeled sentence-pair setup for entailment, contradiction, and neutral classification; the repo's chosen cross-encoder model is documented as being trained on SNLI as well as MultiNLI.

7. Faria, D., Santos, E., Balasubramani, B. S., Silva, M. C., Couto, F. M., and Pesquita, C. (2025). "AgreementMakerLight." Semantic Web.  
   Link: https://journals.sagepub.com/doi/10.3233/SW-233304

   Why it matters here: AML is one of the structural matchers used to populate the repo's `matched` flag. If AML identifies a correspondence, that pair contributes to the binary match signal in the merged CSV.

8. Jimenez-Ruiz, E., and Cuenca Grau, B. (2011). "LogMap: Logic-Based and Scalable Ontology Matching." ISWC 2011.  
   Link: https://doi.org/10.1007/978-3-642-25073-6_18

   Why it matters here: LogMap is the second structural matcher used to populate the repo's `matched` flag. The system is designed for scalable ontology matching with built-in reasoning and diagnosis.

9. Speer, R., Chin, J., and Havasi, C. (2017). "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." AAAI 2017.  
   Link: https://ojs.aaai.org/index.php/AAAI/article/view/11164

   Why it matters here: this is the standard citation for ConceptNet as a general knowledge graph with labeled relations. It supports the repo's use of ConceptNet as an external source for semantic discovery.

10. Haller, A., Janowicz, K., Cox, S., Lefrançois, M., Taylor, K., Le Phuoc, D., and Stadler, C. (2019). "The modular SSN ontology: A joint W3C and OGC standard specifying the semantics of sensors, observations, sampling, and actuation." Semantic Web, 10(1), 9–32.  
    Link: https://doi.org/10.3233/SW-180320  
    W3C Recommendation: https://www.w3.org/TR/vocab-ssn/

    Why it matters here: SSN/SOSA defines a minimal, standardized set of relations for cyber-physical and sensor-driven systems — including `ssn:hasSubSystem`, `sosa:observes`, `sosa:isActedOnBy`, `ssn:implements`, and `sosa:hosts`. These are directly applicable as canonical relation targets when normalizing association names from engineering conceptual models. SSN/SOSA is a W3C + OGC joint standard, making it a citable, domain-neutral authority for relation normalization in systems modelling contexts.

11. Smith, B., Ceusters, W., Klagges, B., Köhler, J., Kumar, A., Lomax, J., Mungall, C., Neuhaus, F., Rector, A. L., and Rosse, C. (2005). "Relations in biomedical ontologies." Genome Biology, 6(5), R46.  
    Link: https://doi.org/10.1186/gb-2005-6-5-r46  
    OBO Foundry RO: https://www.obofoundry.org/ontology/ro.html

    Why it matters here: the OBO Relation Ontology (RO) establishes a formal, minimal set of cross-domain relations grounded in Basic Formal Ontology (BFO): `part_of`, `has_part`, `participates_in`, `has_function`, `located_in`, `causally_upstream_of`, and others. RO is widely adopted in engineering and biomedical ontologies and provides a peer-reviewed, versioned authority for what constitutes a minimal canonical relation set — making it a stronger foundation for relation normalization than ad-hoc choices.

## What The References Validate

### `wup`
The literature supports the idea of Wu-Palmer similarity as a taxonomic score based on the depth of two synsets and their least common subsumer. In NLTK's implementation, the score is derived from WordNet hypernym paths and is intended to be in the range 0 to 1. The repo's version is still compatible with that literature, but the `max_wup` and `avg_wup` blend is a local design choice.

### `cosine_avg`
Sentence-BERT was introduced specifically to produce sentence embeddings that can be compared with cosine similarity. Sentence-Transformers documentation also treats cosine as the default similarity function for embedding comparison. The repo's `cosine_avg` is therefore a standard embedding-similarity measure, although the exact token aggregation used in the repo is its own implementation.

### `entailment_a_covers_b` and `entailment_b_covers_a`
The NLI model card confirms that the cross-encoder outputs three-way scores for contradiction, entailment, and neutral. The academic grounding comes from BERT-style paired sentence classification on NLI benchmarks: SNLI and MultiNLI define the sentence-pair entailment task, and BERT demonstrates that a transformer can jointly encode the premise and hypothesis and classify the relation with a simple output head. That makes the directional scores valid as probabilities that one concept's textual signature entails the other. In ontology terms, a high score in one direction can be read as possible subsumption, while high scores in both directions are more consistent with equivalence or synonymy.

This last step is an inference from NLI semantics and ontology-matching practice, not a direct claim of the model card.

### `entailment_f1`
The repo's `entailment_f1` is not a harmonic F1 score. It is the max of the two directional entailment probabilities. That is a reasonable heuristic for "strongest directional relationship" because subsumption is asymmetric, while synonymy/equivalence is symmetric.

### `matched`
The repo's `matched` column is a binary structural-match signal: it is set when AML or LogMap confirms a correspondence between two entities. That makes it a direct indicator of matcher agreement rather than a learned similarity score.

The literature support here comes from the ontology-matching systems themselves:

- AML provides structural matching, filtering, and alignment validation.
- LogMap provides logic-based ontology matching with reasoning and diagnosis.

So `matched` is best understood as a system output derived from established ontology matchers, not as an independent metric with a separate mathematical definition.

### `coherence_sym`
This metric is not a standard published metric under that exact name. It is a derived topological coherence score built from local neighborhood comparison, combining a WordNet similarity signal (`wup`) with embedding cosine similarity, then symmetrizing the result with a geometric mean.

The literature does support the general idea of ontology-based semantic coherence and neighborhood-based alignment:

- Gurevych et al. (2003), "Semantic Coherence Scoring Using an Ontology"  
  Link: https://aclanthology.org/N03-1012/

- Porzel et al. (2003), "Ontology-based Contextual Coherence Scoring"  
  Link: https://aclanthology.org/W03-2115/

- Ferrandez et al. (2010), "Aligning FrameNet and WordNet based on Semantic Neighborhoods"  
  Link: https://aclanthology.org/L10-1436/

Taken together, these show that using local semantic neighborhoods to judge coherence is well grounded. The exact formula `sqrt(WUP * cosine)` used in this repo is still a repository-specific engineering choice.

### ConceptNet usage
ConceptNet is an open multilingual knowledge graph with labeled edges and external links to other lexical resources. That makes it a good fit for relation discovery among unmatched entities, especially when the repo wants to propose equivalence or subsumption candidates beyond direct WordNet matches.

The repo's use of ConceptNet is consistent with the resource's intended role as a general commonsense and lexical knowledge source, but the exact candidate-selection logic is again a local implementation choice.

### Relation normalization (SSN/SOSA and BFO/RO)
The relation normalization step maps heterogeneous association names from conceptual models to a canonical relation set. SSN/SOSA and BFO/RO provide two independent, citable authorities for what that canonical set should contain:

- SSN/SOSA supplies relations tuned to physical systems and sensor networks: structural containment (`hasSubSystem`), observation (`observes`), actuation (`isActedOnBy`), and implementation (`implements`).
- BFO/RO supplies cross-domain mereological and causal relations: `part_of`, `has_function`, `participates_in`, `causally_upstream_of`, and `located_in`.

Together they cover the semantic space of the engineering models in this repo without depending on ad-hoc choices. The normalization algorithm then uses Wu-Palmer similarity (ref. 1) against representative WordNet lemmas for each canonical relation to score and assign each observed association name.

## Bottom Line

The repo's metric stack is a mix of:

- standard similarity models and ontology matchers from the literature (`wup`, cosine sentence similarity, NLI entailment, ConceptNet, AML, LogMap)
- and local derived aggregations (`max_wup + avg_wup`, `coherence_sym`, `entailment_f1`)

So the literature validates the ingredients, while the repo defines the final composed metrics.
