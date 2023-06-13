from langchain.schema import SystemMessage, HumanMessage, AIMessage


def extract_memory(messages):
    """
    extract memory from messages
    :param messages:
    :return:
    """
    if len(messages) == 0:
        raise ValueError("session messages are empty")

    new_message = messages[-1]
    if new_message['role'] != 'user':
        raise ValueError("last message is not human message")
    # TODO how to pass the system message to the chain ?
    system_message = messages[0]
    history_messages = messages[1:-1]
    history = []
    history.append(SystemMessage(content=system_message['content']))
    for query, ans in zip(history_messages[0::2], history_messages[1::2]):
        assert query['role'] == 'user'
        assert ans['role'] == 'assistant'
        history.append(HumanMessage(content=query['content']))
        history.append(AIMessage(content=ans['content']))
    return new_message['content'], history