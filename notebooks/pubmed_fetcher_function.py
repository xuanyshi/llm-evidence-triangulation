import datetime
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List

__all__ = ['crawler', 'fetch_pubmed']


def pmid_splitter(pmid_list: List[str], chunk_size: int = 300) -> List[List[str]]:
    """Split a large PMID list into chunks of `chunk_size` for NCBI requests."""
    chunks = [pmid_list[i:i + chunk_size] for i in range(0, len(pmid_list), chunk_size)]
    return chunks


def fetch_pubmed(pmid_list: List[str], chunk_size: int = 50) -> pd.DataFrame:
    """Fetch PubMed metadata for a list of PMIDs.

    Parameters
    ----------
    pmid_list : list[str]
        A list of PubMed IDs to retrieve.
    chunk_size : int, default 50
        Number of PMIDs to query per API request to stay within NCBI limits.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing one row per PMID with parsed metadata.
    """
    pmid_chunks = pmid_splitter(pmid_list, chunk_size)
    return crawler(pmid_chunks)


#https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=16570560
def crawler(pmid_chunks):

    retmode = "xml"

    rettype = ""

    full_df_list = []
    count = 0
    for chunk in pmid_chunks:

        print(f'chunk #{count} starts at:',datetime.datetime.now())

        chunk_str = str(chunk).replace("'",'').replace('[','').replace(']','').replace(' ','')
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "retmode": retmode,
                "id": chunk_str,
                "rettype": rettype,
            },
            timeout=30,
        )
        if not resp.ok:
            print(f"Request failed for chunk #{count} (status {resp.status_code}) – skipping")
            count += 1
            time.sleep(1)
            continue

        try:
            root = ET.fromstring(resp.text.encode("utf-8"))
        except ET.ParseError as e:
            print(f"XML parse error for chunk #{count}: {e} – skipping")
            count += 1
            time.sleep(1)
            continue

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

                background = None
                conclusion = None
                result = None
                method = None
                label_conclusion = None
                label_result = None

                article_title = art_info.find('.//ArticleTitle').text

                p_type = art_info.findall('Abstract/')

                a = [i  for i in p_type if (i.tag == 'AbstractText')]

                full_abs = [i.text for i in a]
                full_abs = ''.join([i.text for i in a if i.text is not None])
                info_dict['full_abs'] = full_abs

                for i in a:

                  ##getting method
                  if 'NlmCategory' in i.attrib.keys():
                      if i.attrib['NlmCategory'] == 'BACKGROUND':
                          background = i.text
                      else:
                          pass
                  else:
                      background = None
                  ##gettomg conclusion
                  if 'NlmCategory' in i.attrib.keys():
                      if i.attrib['NlmCategory'] == 'CONCLUSIONS':
                          conclusion = i.text
                      else:
                          pass
                  else:
                      conclusion = None

                  if 'NlmCategory' in i.attrib.keys():
                      if i.attrib['NlmCategory'] == 'RESULTS':
                          result = i.text
                      else:
                          pass
                  else:
                      result = None

                  if 'NlmCategory' in i.attrib.keys():
                      if i.attrib['NlmCategory'] == 'METHODS':
                          method = i.text
                      else:
                          pass
                  else:
                      method = None

                  if 'Label' in i.attrib.keys():
                      if 'conclusion' in i.attrib['Label'].lower():
                          label_conclusion = i.text
                      else:
                          pass
                  else:
                      label_conclusion = None

                  if 'Label' in i.attrib.keys():
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
                info_dict['# of comments'] = comment_count
        #################################################### mesh info
            mesh_tree = article.findall('MeshHeadingList')
            for mesh_info in mesh_tree:

                all_mesh_list = []
                major_mesh_list = []
                for i in mesh_info:

                    full_mesh = ''
                    major_mesh = ''

                    if_major = 0

                    for m in i:

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

                    all_mesh_list.append(full_mesh)
                    major_mesh_list.append(major_mesh)
                    major_mesh_list = [value for value in major_mesh_list if '*' in value]


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

    if not full_df_list:
        return pd.DataFrame()

    final_df = pd.concat(full_df_list)
    return final_df