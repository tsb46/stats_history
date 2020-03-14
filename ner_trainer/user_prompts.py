entity_question = [
    {
        'type': 'input',
        'name': 'entity_check',
        'message': 'Is there an entity in this document? Enter "y" for yes,'
                   '"n" for no. Enter "p" if you would like a report of your progress.'
                   'Enter "s" to save your current progress'
    }
]

overwrite_pickle = [
    {
        'type': 'confirm',
        'name': 'overwrite_pickle',
        'message': 'there is an existing "training" pickle '
                   'file in your output. Would you like to continue '
                   'appending to that?'
    }
]


