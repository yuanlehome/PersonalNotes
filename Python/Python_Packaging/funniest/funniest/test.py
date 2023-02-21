from markdown import markdown

def joke():
    return markdown(u'How do you tell HTML from HTML5?'
                    u'Try it out in **Internet Explorer**.'
                    u'Does it work?'
                    u'No?'
                    u'It\'s HTML5.')