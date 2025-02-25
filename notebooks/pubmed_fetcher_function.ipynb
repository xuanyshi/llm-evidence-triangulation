#https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=16570560
def crawler(pmid_chunks):

    retmode = "xml"

    rettype = ""

    full_df_list = []
    count = 0
    for chunk in pmid_chunks:

        print(f'chunk #{count} starts at:',datetime.datetime.now())

        chunk_str = str(chunk).replace("'",'').replace('[','').replace(']','').replace(' ','')
        re = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=" + retmode + "&id=" + chunk_str + "&rettype=" + rettype)
        #time.sleep(3)
        root = ET.fromstring(re.text.encode("utf-8"))

        ## getting data from each pmid chunk

        full_dict = {}
        for article in root.findall('PubmedArticle/MedlineCitation'):
            pmid = article[0].text
            info_dict = {}


        #############################################################  article info
            article_tree = article.findall('Article')

            for art_info in article_tree:

                if art_info.find('.//PubDate'):
                    pub_date = art_info.find('.//PubDate')
                    year_element = pub_date.find('Year')
                    month_element = pub_date.find('Month')
                    year = year_element.text if year_element is not None else None
                    month = month_element.text if month_element is not None else None
                else:
                    year = None
                    month = None

                info_dict['pub_year'] = year
                info_dict['pub_month'] = month

                if art_info.find('.//ArticleDate[@DateType="Electronic"]'):
                    epub_date = art_info.find('.//ArticleDate[@DateType="Electronic"]')
                    epub_year_element = epub_date.find('Year')
                    epub_month_element = epub_date.find('Month')
                    epub_year = epub_year_element.text if epub_year_element is not None else None
                    epub_month = epub_month_element.text if epub_month_element is not None else None
                else:
                    epub_year = None
                    epub_month = None

                publication_types = [pub_type.text for pub_type in art_info.findall('.//PublicationType')]

                info_dict['pub_year'] = year
                info_dict['pub_month'] = month
                info_dict['epub_year'] = epub_year
                info_dict['epub_month'] = epub_month
                info_dict['pub_types'] = publication_types
                # if art_info.find('.//PubDate'):
                #   pub_date = art_info.find('.//PubDate')
                #   year = pub_date.find('Year').text if pub_date.find('Year').text else None
                #   month = pub_date.find('Month').text if pub_date.find('Month').text else None
                # else:
                #   year = None
                #   month = None

                # info_dict['pub_year'] = year
                # info_dict['pub_month'] = month

                background = None
                conclusion = None
                result = None
                method = None
                label_conclusion = None
                label_result = None

                article_title = art_info.find('.//ArticleTitle').text


                p_type = art_info.findall('Abstract/')
                #p_type = [i.attrib  for i in p_type ]

                a = [i  for i in p_type if (i.tag == 'AbstractText')]
                #a = [i for i in a if i!= None]

                full_abs = [i.text for i in a]
                full_abs = ''.join([i.text for i in a if i.text is not None])
                info_dict['full_abs'] = full_abs

                for i in a:

                  ##getting method
                  if 'NlmCategory' in i.attrib.keys():
                      #print(i.attrib['NlmCategory'])
                      if i.attrib['NlmCategory'] == 'BACKGROUND':
                          background = i.text
                      else:
                          pass
                  else:
                      background = None
                  ##gettomg conclusion
                  if 'NlmCategory' in i.attrib.keys():
                      #print(i.attrib['NlmCategory'])
                      if i.attrib['NlmCategory'] == 'CONCLUSIONS':
                          conclusion = i.text
                      else:
                          pass
                  else:
                      conclusion = None

                  if 'NlmCategory' in i.attrib.keys():
                      #print(i.attrib['NlmCategory'])
                      if i.attrib['NlmCategory'] == 'RESULTS':
                          result = i.text
                      else:
                          pass
                  else:
                      result = None

                  if 'NlmCategory' in i.attrib.keys():
                      #print(i.attrib['NlmCategory'])
                      if i.attrib['NlmCategory'] == 'METHODS':
                          method = i.text
                      else:
                          pass
                  else:
                      method = None

                  if 'Label' in i.attrib.keys():
                      #print(i.attrib['NlmCategory'])
                      if 'conclusion' in i.attrib['Label'].lower():
                          label_conclusion = i.text
                      else:
                          pass
                  else:
                      label_conclusion = None

                  if 'Label' in i.attrib.keys():
                      #print(i.attrib['NlmCategory'])
                      if 'result' in i.attrib['Label'].lower():
                          label_result = i.text
                      else:
                          pass
                  else:
                      label_result = None

                info_dict['article_title'] = article_title
                info_dict['background'] = background
                info_dict['method'] = method
                info_dict['result'] = result
                info_dict['conclusion'] = conclusion
                info_dict['label_result'] = label_result
                info_dict['label_conclusion'] = label_conclusion
        #################################################### commentaries info
            comment_tree = article.findall('CommentsCorrectionsList')
            for comment_info in comment_tree:
                comment_count = 0
                for i in comment_info:
                    if i.attrib['RefType'] == 'CommentIn':
                        comment_count+=1
                        #print([x.text for x in i][0])
                #print(comment_count)
                #omment = comment_info.findall('RefSource')
                #p_type = [i.attrib  for i in p_type ]

                #a = [i  for i in p_type if (i.tag == 'AbstractText')]
                #a = [i for i in a if i!= None]
                #print(comment)
        #         for i in a:
        #             #print(i.attrib.keys())
        #             if 'NlmCategory' in i.attrib.keys():
        #                 p_type = i.text
        #             else:
        #                 p_type = None
        #         print(p_type)
        #         info_dict['comment'] = p_type
                info_dict['# of comments'] = comment_count
        #################################################### mesh info
            mesh_tree = article.findall('MeshHeadingList')
            for mesh_info in mesh_tree:
        #         mesh = mesh_info.findall('MeshHeading/')
        #         all_mesh = [i.text for i in mesh]

        #         mesh_tags = mesh_info.findall('MeshHeading/*[@MajorTopicYN="Y"]')
        #         mesh_tags = [i.text for i in mesh_tags]

                all_mesh_list = []
                major_mesh_list = []
                for i in mesh_info:

                    full_mesh = ''
                    major_mesh = ''

                    if_major = 0

                    for m in i:
        #                 if m.attrib['MajorTopicYN']=='Y':
        #                     major_mesh += m.text
        #                     major_mesh +='/'


                        if m.tag=='DescriptorName':
                            descriptor=m.text
                            qualifier = ''
                            full_mesh+=descriptor
                            full_mesh += ''
                            full_mesh+=qualifier

                            major_mesh+=descriptor

                            major_mesh += ''
                            if m.attrib['MajorTopicYN'] == 'Y':
                                major_mesh +='*'
                            major_mesh+=qualifier

                        elif m.tag=='QualifierName':
                            descriptor=''
                            qualifier = m.text

                            full_mesh+=descriptor
                            full_mesh += '/'
                            full_mesh+=qualifier

                            if m.attrib['MajorTopicYN'] == 'Y':

                                major_mesh+=descriptor
                                major_mesh += '/'
                                major_mesh+=qualifier
                                major_mesh+='*'
                         ## combining sub mesh and main mesh

                    #full_mesh = full_mesh[:-1]
                    #major_mesh = major_mesh[:-1]


                    all_mesh_list.append(full_mesh)
                    major_mesh_list.append(major_mesh)
                    major_mesh_list = [value for value in major_mesh_list if '*' in value]

        #         desp = mesh_info.findall('MeshHeading/')
        #         for i in desp:
        #             print(i.tag,'\n')



                info_dict['all_mesh_terms'] = all_mesh_list
                info_dict['major_mesh_terms'] = major_mesh_list
        ##############################################################
            full_dict[pmid] = info_dict
            info_df = pd.DataFrame(full_dict).transpose().reset_index()

        info_df = pd.DataFrame(full_dict).transpose().reset_index()
        full_df_list.append(info_df)
        print(f'chunk #{count} is done at:',datetime.datetime.now())
        count+=1
        print('\n')
        time.sleep(5)

    final_df = pd.concat(full_df_list)
    return final_df

def pmid_splitter(pmid):
    query_split = [pmid[i:i + 300] for i in range(0, len(pmid), 300)]
    print(len(query_split))
    return query_split

pmid_chunks = pmid_splitter(pmid_all) ## input a list of wanted pmids
all_got_df = crawler(pmid_chunks).rename(columns={'index':'pmid'})
