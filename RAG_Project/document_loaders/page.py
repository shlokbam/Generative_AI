from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/?afid=p240%7Cgo~cmp-11182149775~adg-109263621973~ad-784581523341_kwd-987393509~dev-c~ext-~prd-~mca-~nt-search&cid=aos-in-kwgo-txt-mac-mac--"

data = WebBaseLoader(url)

documents = data.load()
print(documents[0].page_content)
