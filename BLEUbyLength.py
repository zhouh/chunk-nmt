'''
Translates a source file using a translation model.
'''
import argparse

import subprocess
import re
import os


def getBLEU():

    return 0



def main(bleu_scrip,
         valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok',
                         '../data/dev/newstest2011.fr.tok'],
         length=10):

                len_chose = [10, 20, 30, 40, 50]


                source_file = open(valid_datasets[0], 'r')

                target_file = open(valid_datasets[1], 'r')

                reference0_file = open(valid_datasets[2]+'0', 'r')
                reference1_file = open(valid_datasets[2]+'1', 'r')
                reference2_file = open(valid_datasets[2]+'2', 'r')
                reference3_file = open(valid_datasets[2]+'3', 'r')

                source_sents = source_file.readlines()
                target_sents = target_file.readlines()
                reference0_sents = reference0_file.readlines()
                reference1_sents = reference1_file.readlines()
                reference2_sents = reference2_file.readlines()
                reference3_sents = reference3_file.readlines()

                idx_set=[ [], [], [], [], [], [] ]

                target_set = [ [] ] * len(idx_set)
                r0_set = [ [] ] * len(idx_set)
                r1_set = [ [] ] * len(idx_set)
                r2_set = [ [] ] * len(idx_set)
                r3_set = [ [] ] * len(idx_set)



                for idx, sent in enumerate(source_sents):
                    tokens = sent.strip().split()
                    l = len(tokens)
                    # print l

                    if l <= len_chose[0]:
                        idx_set[0].append(idx)
                        # print '< 10', l, idx
                    elif l <= len_chose[1]:
                        idx_set[1].append(idx)
                    elif l <= len_chose[2]:
                        idx_set[2].append(idx)
                    elif l <= len_chose[3]:
                        idx_set[3].append(idx)
                    elif l <= len_chose[4]:
                        idx_set[4].append(idx)
                    else:
                         idx_set[5].append(idx)




                # get the filter of the sentences
                for i in range(6):

                    # print idx_set[i]
                    target_set[i] = [target_sents[k].strip() for k in idx_set[i]]
                    r0_set[i] = [reference0_sents[k].strip() for k in idx_set[i]]
                    r1_set[i] = [reference1_sents[k].strip() for k in idx_set[i]]
                    r2_set[i] = [reference2_sents[k].strip() for k in idx_set[i]]
                    r3_set[i] = [reference3_sents[k].strip() for k in idx_set[i]]

                    with open('./translate.tmp', 'w') as f:
                        print >> f , '\n'.join(target_set[i])

                    with open('./tmp.reference0', 'w') as f:
                        print >> f, '\n'.join(r0_set[i])

                    with open('./tmp.reference1', 'w') as f:
                        print >> f, '\n'.join(r1_set[i])

                    with open('./tmp.reference2', 'w') as f:
                        print >> f, '\n'.join(r2_set[i])

                    with open('./tmp.reference3', 'w') as f:
                        print >> f, '\n'.join(r3_set[i])


                    cmd_bleu_cmd = ['perl', bleu_scrip, \
                                     './tmp.reference', \
                                     '<', \
                                     './translate.tmp', \
                                     '>'
                                     './output.eva']

                    subprocess.check_call(" ".join(cmd_bleu_cmd), shell=True)

                    fin = open('./output.eva', 'rU')
                    out = re.search('BLEU = [-.0-9]+', fin.readlines()[0])
                    fin.close()

                    bleu_score = float(out.group()[7:])

                    if i < len(len_chose):
                        print '### Len <= ', len_chose[i]
                    else:
                        print '### Len > ', len_chose[i-1]
                    print '### BLEU:', bleu_score, 'total: ', len(target_set[i])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-length', type=int, default=10)
    parser.add_argument('bleu_scrip', type=str)
    parser.add_argument('valid_source', type=str)
    parser.add_argument('valid_target', type=str)
    parser.add_argument('valid_reference', type=str)


    args = parser.parse_args()

    main(args.bleu_scrip,
         valid_datasets=[args.valid_source, args.valid_target, args.valid_reference],
         length=args.length)

